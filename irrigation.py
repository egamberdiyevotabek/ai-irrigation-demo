"""
AI Irrigation prototype
- FAO-56 Penman-Monteith ET0 calculation
- Simple soil water balance and irrigation decision
- ML model (RandomForest) to predict soil moisture from features
- Simulated sensors and actuator interfaces (replace with real I/O)
 
Requirements:
pip install numpy pandas scikit-learn joblib
"""
 
import math
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load
from datetime import datetime, timedelta
 
# --------------------------
# CONFIG
# --------------------------
CROP_COEFFICIENT = {
   # example Ks by stage for cotton (adapt as needed)
   "initial": 0.3,
   "development": 0.6,
   "mid": 1.15,
   "late": 0.8
}
 
SOIL_FIELD_CAPACITY = 0.30   # m3/m3 (volumetric)
SOIL_WILTING_POINT = 0.12   # m3/m3
ROOT_ZONE_DEPTH = 0.9       # m (user-specified)
SAFE_SOIL_MOISTURE = 0.6    # fraction of available water (0-1), threshold to trigger irrigation
IRRIGATION_EFFICIENCY = 0.75  # distribution & application efficiency
 
MODEL_PATH = "sm_predictor.joblib"
 
# --------------------------
# FAO-56 Penman-Monteith (reference ET0)
# --------------------------
def et0_penman_monteith(t_mean_c, t_min_c, t_max_c, rh_mean, wind_speed_2m, solar_rad, elevation=0, lat=41.3, doy=None):
   """
   Approximate FAO-56 Penman-Monteith implementation.
   Inputs:
       t_mean_c, t_min_c, t_max_c: °C
       rh_mean: relative humidity (0-100)
       wind_speed_2m: m/s at 2 m
       solar_rad: MJ/m2/day (incoming shortwave)
       elevation: m
       lat: degrees (used for clear-sky calc if needed)
       doy: day of year (1-365)
   Returns:
       ET0 in mm/day
   Note: This is a practical implementation for prototyping. For production use, cross-check with FAO-56 references.
   """
   if doy is None:
       doy = datetime.utcnow().timetuple().tm_yday
 
   # Constants
   G_sc = 0.0820  # solar constant MJ m-2 min-1
   sigma = 4.903e-9  # Stefan-Boltzmann constant MJ K-4 m-2 day-1
   gamma = 0.665e-3 * 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26  # psychrometric constant kPa/°C
 
   t_mean_k = t_mean_c + 273.16
 
   # Saturation vapor pressure es(T) using Tetens
   def es_temp(t):
       return 0.6108 * math.exp((17.27 * t) / (t + 237.3))
   es_tmin = es_temp(t_min_c)
   es_tmax = es_temp(t_max_c)
   es = (es_tmin + es_tmax) / 2.0  # kPa
 
   ea = es * (rh_mean / 100.0)  # actual vapor pressure kPa
 
   # slope of saturation vapor pressure curve (kPa/°C)
   delta = (4098 * (0.6108 * math.exp((17.27 * t_mean_c) / (t_mean_c + 237.3)))) / ((t_mean_c + 237.3) ** 2)
 
   # Extraterrestrial radiation Ra (MJ/m2/day) — approximate using day angle
   dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
   solar_decl = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
   lat_rad = math.radians(lat)
   ws = math.acos(-math.tan(lat_rad) * math.tan(solar_decl))
   Ra = (24 * 60 / math.pi) * G_sc * dr * (ws * math.sin(lat_rad) * math.sin(solar_decl) + math.cos(lat_rad) * math.cos(solar_decl) * math.sin(ws))
   # Clear-sky radiation
   Rso = (0.75 + 2e-5 * elevation) * Ra
 
   # Net shortwave radiation
   albedo = 0.23
   Rns = (1 - albedo) * solar_rad
 
   # Net outgoing longwave radiation (approx)
   tmax_k = t_max_c + 273.16
   tmin_k = t_min_c + 273.16
   Rnl = sigma * ((tmax_k ** 4 + tmin_k ** 4) / 2) * (0.34 - 0.14 * math.sqrt(ea)) * (1.35 * min(solar_rad / Rso, 1.0) - 0.35)
 
   Rn = Rns - Rnl  # net radiation MJ/m2/day
 
   # Soil heat flux G ~ 0 for daily time-step
   G = 0
 
   # FAO Penman-Monteith
   numerator = 0.408 * delta * (Rn - G) + gamma * (900.0 / (t_mean_c + 273.0)) * wind_speed_2m * (es - ea)
   denominator = delta + gamma * (1 + 0.34 * wind_speed_2m)
   et0 = numerator / denominator  # mm/day
 
   if et0 < 0:
       et0 = 0.0
   return et0
 
# --------------------------
# Crop ETc
# --------------------------
def crop_evapotranspiration(et0, crop_stage="mid"):
   kc = CROP_COEFFICIENT.get(crop_stage, 1.0)
   return et0 * kc  # mm/day
 
# --------------------------
# Soil water balance and irrigation decision
# --------------------------
def volumetric_to_depth(vwc, root_depth_m):
   """Convert volumetric water content (m3/m3) to depth (mm) of water in root zone."""
   return vwc * root_depth_m * 1000.0  # mm
 
def depth_to_volumetric(depth_mm, root_depth_m):
   return depth_mm / (root_depth_m * 1000.0)
 
def compute_available_water(field_capacity, wilting_point, root_depth_m):
   aw = (field_capacity - wilting_point) * root_depth_m * 1000.0  # mm
   return max(aw, 0.0)
 
def irrigation_needed(current_vwc, field_capacity, wilting_point, root_depth_m, safe_fraction=SAFE_SOIL_MOISTURE):
   """
   Decide whether irrigation is needed based on fraction of available water.
   Returns (bool, deficit_mm)
   """
   current_depth = volumetric_to_depth(current_vwc, root_depth_m)
   aw = compute_available_water(field_capacity, wilting_point, root_depth_m)  # mm
   available_now = current_depth - volumetric_to_depth(wilting_point, root_depth_m)
   fraction = available_now / aw if aw > 0 else 0.0
 
   if fraction <= safe_fraction:
       # aim to refill up to field capacity (or up to fraction e.g., 0.9 of AW)
       target_depth = volumetric_to_depth(field_capacity, root_depth_m)
       deficit = max(0.0, target_depth - current_depth)
       # account for efficiency
       required_irrigation = deficit / IRRIGATION_EFFICIENCY
       return True, required_irrigation  # mm to apply
   return False, 0.0
 
# --------------------------
# Simple ML model to predict soil moisture
# --------------------------
def make_synthetic_training_data(n=2000, seed=42):
   """
   Create synthetic dataset with features:
   - t_mean, t_min, t_max, rh_mean, wind, solar, precipitation, previous_vwc, days_since_irrig
   Target: next_day_vwc (volumetric water content)
   This is for prototype training only — replace with your real sensor historical data.
   """
   np.random.seed(seed)
   t_mean = np.random.normal(25, 6, n)
   t_min = t_mean - np.random.uniform(3, 8, n)
   t_max = t_mean + np.random.uniform(3, 8, n)
   rh = np.clip(np.random.normal(60, 15, n), 10, 100)
   wind = np.abs(np.random.normal(2.0, 1.0, n))
   solar = np.clip(np.random.normal(18, 6, n), 1, 30)  # MJ/m2/day
   precip = np.clip(np.random.exponential(2.0, n), 0, 30)  # mm
   prev_vwc = np.clip(np.random.normal(0.22, 0.06, n), 0.08, 0.45)
   days_since_irrig = np.random.poisson(3, n)
 
   # compute synthetic ET0 and ETc and then update vwc
   et0 = np.array([et0_penman_monteith(t_mean[i], t_min[i], t_max[i], rh[i], wind[i], solar[i]) for i in range(n)])
   etc = et0 * 0.8  # assume crop coefficient 0.8 on average
   # water loss = ETc - precip (converted to volumetric by dividing by root depth*1000)
   delta_depth = etc - precip  # mm (positive means loss)
   delta_vwc = -delta_depth / (ROOT_ZONE_DEPTH * 1000.0)  # negative lowers vwc
   next_vwc = np.clip(prev_vwc + delta_vwc + np.random.normal(0, 0.01, n), 0.05, 0.50)
 
   df = pd.DataFrame({
       "t_mean": t_mean,
       "t_min": t_min,
       "t_max": t_max,
       "rh": rh,
       "wind": wind,
       "solar": solar,
       "precip": precip,
       "prev_vwc": prev_vwc,
       "days_since_irrig": days_since_irrig,
       "next_vwc": next_vwc
   })
   return df
 
def train_sm_predictor(df, save_path=MODEL_PATH):
   features = ["t_mean", "t_min", "t_max", "rh", "wind", "solar", "precip", "prev_vwc", "days_since_irrig"]
   X = df[features].values
   y = df["next_vwc"].values
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
   model = RandomForestRegressor(n_estimators=200, random_state=1)
   model.fit(X_train, y_train)
   print("Model trained. Test R2:", model.score(X_test, y_test))
   dump(model, save_path)
   return model
 
def load_sm_predictor(path=MODEL_PATH):
   try:
       model = load(path)
       return model
   except Exception as e:
       print("Model load failed:", e)
       return None
 
# --------------------------
# Sensor & actuator interfaces (to replace with real hardware code)
# --------------------------
class SimulatedSensors:
   def __init__(self):
       self.vwc = 0.25  # initial volumetric soil moisture
       self.last_irrigation = None
 
   def read_environment(self):
       # Return dictionary of env measurements (these should be replaced by real sensors)
       # For prototype we generate small random variations around typical values
       now = datetime.utcnow()
       t_mean = 25 + np.random.normal(0, 2)
       t_min = t_mean - (3 + np.random.rand()*4)
       t_max = t_mean + (3 + np.random.rand()*4)
       rh = np.clip(60 + np.random.normal(0, 8), 10, 100)
       wind = abs(2.0 + np.random.normal(0, 0.7))
       solar = np.clip(18 + np.random.normal(0, 4), 0, 30)
       precip = np.random.choice([0.0, 0.0, 0.0, np.random.exponential(3.0)])  # mostly no rain
       return {
           "datetime": now,
           "t_mean": t_mean,
           "t_min": t_min,
           "t_max": t_max,
           "rh": rh,
           "wind": wind,
           "solar": solar,
           "precip": precip,
           "vwc": self.vwc
       }
 
   def apply_irrigation(self, depth_mm):
       # applying water increases vwc; this should be replaced with actuator control + measured feedback
       added_vwc = (depth_mm * IRRIGATION_EFFICIENCY) / (ROOT_ZONE_DEPTH * 1000.0)
       self.vwc = min(self.vwc + added_vwc, SOIL_FIELD_CAPACITY)
       self.last_irrigation = datetime.utcnow()
       print(f"[SIM] Applied irrigation: {depth_mm:.1f} mm -> vwc now {self.vwc:.3f}")
 
   def natural_update(self, etc_mm, precip_mm):
       # update vwc based on ETc and precipitation for the day (used between loops)
       delta_depth = etc_mm - precip_mm
       delta_vwc = -delta_depth / (ROOT_ZONE_DEPTH * 1000.0)
       self.vwc = np.clip(self.vwc + delta_vwc, 0.02, 0.5)
 
# --------------------------
# Main control loop prototype
# --------------------------
def control_loop(iterations=10, train_model_first=True):
   sensors = SimulatedSensors()
   model = None
 
   # Train model on synthetic data if needed (or load existing)
   if train_model_first:
       print("Training synthetic ML predictor...")
       df = make_synthetic_training_data(n=1500)
       model = train_sm_predictor(df)
   else:
       model = load_sm_predictor()
       if model is None:
           print("No model available, training new model from synthetic data.")
           df = make_synthetic_training_data(n=1500)
           model = train_sm_predictor(df)
 
   for it in range(iterations):
       data = sensors.read_environment()
       et0 = et0_penman_monteith(data["t_mean"], data["t_min"], data["t_max"], data["rh"], data["wind"], data["solar"])
       etc = crop_evapotranspiration(et0, crop_stage="mid")  # adjust stage as required
 
       print(f"\n[{it+1}/{iterations}] {data['datetime'].isoformat()}  T={data['t_mean']:.1f}C  ET0={et0:.2f} mm  ETc={etc:.2f} mm  vwc={data['vwc']:.3f}")
 
       # ML prediction of next-day VWC
       features = np.array([[data["t_mean"], data["t_min"], data["t_max"], data["rh"], data["wind"], data["solar"], data["precip"], data["vwc"], (0 if sensors.last_irrigation is None else (datetime.utcnow() - sensors.last_irrigation).days)]])
       pred_vwc = model.predict(features)[0]
       print(f"Predicted next-day VWC: {pred_vwc:.3f}")
 
       # Decision based on current measured vwc
       need, required_mm = irrigation_needed(data["vwc"], SOIL_FIELD_CAPACITY, SOIL_WILTING_POINT, ROOT_ZONE_DEPTH)
       if need:
           print(f"IRRIGATION DECISION: Need irrigation. Apply ~{required_mm:.1f} mm (accounting for efficiency).")
           # In real system: actuator.on_for_volume(required_mm) or similar
           sensors.apply_irrigation(required_mm)
       else:
           print("No irrigation needed now.")
 
       # Simulate natural daily update (ETc removes water, precip adds)
       sensors.natural_update(etc_mm=etc, precip_mm=data["precip"])
 
       # Sleep in prototype — in real system use scheduling (cron) or event-based triggers
       time.sleep(0.1)
 
   # Save model at end
   dump(model, MODEL_PATH)
   print("Control loop finished. Model saved to", MODEL_PATH)
 
# --------------------------
# CLI run
# --------------------------
if __name__ == "__main__":
   print("AI Irrigation prototype starting...")
   control_loop(iterations=20, train_model_first=True)