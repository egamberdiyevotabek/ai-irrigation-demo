# AI Irrigation

Autonomous irrigation control system powered by FAO-56 evapotranspiration and machine learning.  
Built for irrigated agriculture (with a focus on Uzbekistan) to reduce water use, improve yield, and prevent soil salinization.

---

## Overview

AI Irrigation collects real-time soil + climate data, calculates crop water demand using FAO-56 Penman–Monteith ET₀, maintains a root-zone soil water balance, and decides when/how much to irrigate.  
In autonomous mode, decisions can be sent directly to pumps/valves without human input.

---

## Goals

- Reduce irrigation water consumption
- Increase crop productivity
- Lower soil salinity risk
- Enable fully autonomous, data-driven irrigation

---

## Key Features

- FAO-56 ET₀ calculation (Penman–Monteith reference evapotranspiration)
- Crop ETc estimation using crop coefficients (Kc by growth stage)
- Root-zone soil water balance
- Irrigation decision engine based on fraction of available water
- ML soil moisture predictor (RandomForestRegressor)
- Sensor & actuator interfaces
  - Prototype currently uses simulated I/O
  - Can be replaced with real hardware (LoRaWAN / MQTT / Modbus / PLC, etc.)

---

## System Architecture (High Level)

1. Sensors measure:
   - Soil moisture (VWC), temperature, EC, pH
   - Climate: temperature, humidity, wind speed, solar radiation, precipitation
   - Groundwater level (optional)
2. Data pipeline cleans and stores measurements
3. FAO-56 + AI module computes ET₀, ETc, predicts moisture, updates water balance
4. Decision module outputs irrigation timing + depth
5. Actuators execute irrigation commands
6. User dashboard/app shows live monitoring and reports

---

## Expected Impact (Targets)

- Water savings: ≥ 45%  
- Yield increase: ≥ 25%  
- Irrigation efficiency (η): ≥ 0.90  
- AI decision accuracy: ≥ 95%

---

## Prototype Code

This repository contains a Python prototype that includes:

- Daily ET₀ (FAO-56 Penman–Monteith)
- Crop ETc calculation
- Simple soil water balance in the root zone
- Irrigation trigger/refill logic
- RandomForest model to predict next-day soil moisture
- Simulated sensors and irrigation actuator

### Requirements

```bash
pip install numpy pandas scikit-learn joblib
