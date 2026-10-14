"""NexusFlow AI demo API. Uses synthetic data; no live traffic feed is connected."""
from datetime import datetime, timezone
from typing import Literal
import random
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="NexusFlow AI API", version="1.0.0", description="Demo API for predictive traffic intelligence")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
ROADS = [
    {"id": "R01", "name": "Avinashi Road", "zone": "Central", "congestion": 74, "speed_kmh": 19, "vehicles_per_hour": 1840},
    {"id": "R02", "name": "Trichy Road", "zone": "South", "congestion": 42, "speed_kmh": 34, "vehicles_per_hour": 970},
    {"id": "R03", "name": "Mettupalayam Road", "zone": "North", "congestion": 28, "speed_kmh": 46, "vehicles_per_hour": 720},
    {"id": "R04", "name": "Sathy Road", "zone": "East", "congestion": 86, "speed_kmh": 12, "vehicles_per_hour": 2210},
    {"id": "R05", "name": "Race Course", "zone": "Central", "congestion": 51, "speed_kmh": 29, "vehicles_per_hour": 1120},
    {"id": "R06", "name": "Pollachi Road", "zone": "South", "congestion": 34, "speed_kmh": 39, "vehicles_per_hour": 810},
]

class PredictionRequest(BaseModel):
    road: str = Field(default="Avinashi Road", min_length=2)
    horizon_minutes: int = Field(default=120, ge=15, le=360)
    weather: Literal["clear", "rain", "heavy_rain"] = "clear"
    event_factor: float = Field(default=1.0, ge=0.7, le=1.5)

@app.get("/")
def root():
    return {"name": "NexusFlow AI API", "status": "operational", "mode": "demo", "docs": "/docs"}

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat(), "data_mode": "synthetic_demo"}

@app.get("/api/traffic")
def traffic():
    """Return sample corridor readings with small reproducible-looking jitter."""
    rng = random.Random(datetime.now(timezone.utc).minute)
    result = []
    for road in ROADS:
        level = int(np.clip(road["congestion"] + rng.randint(-3, 3), 0, 100))
        result.append({**road, "congestion": level, "status": "Severe" if level >= 80 else "Heavy" if level >= 65 else "Moderate" if level >= 40 else "Low"})
    frame = pd.DataFrame(result)
    return {"items": frame.to_dict(orient="records"), "count": len(result), "data_mode": "synthetic_demo"}

@app.post("/api/predict")
def predict(req: PredictionRequest):
    road = next((r for r in ROADS if r["name"].lower() == req.road.lower()), None)
    if road is None:
        return {"error": "Unknown road", "available_roads": [r["name"] for r in ROADS]}
    horizon_factor = req.horizon_minutes / 120
    weather_factor = {"clear": 0, "rain": 7, "heavy_rain": 15}[req.weather]
    # A transparent baseline formula for the prototype, not a trained forecasting model.
    predicted = int(np.clip((road["congestion"] + 11 * horizon_factor + weather_factor) * req.event_factor, 0, 100))
    speed = max(5, int(road["speed_kmh"] * (1 - max(0, predicted - road["congestion"]) / 150)))
    action = "Consider alternate routing and emergency-priority review." if predicted >= 75 else "Monitor queues and adjust signal timing if needed." if predicted >= 45 else "Maintain current signal plan and continue monitoring."
    return {"road": road["name"], "horizon_minutes": req.horizon_minutes, "predicted_congestion": predicted, "estimated_speed_kmh": speed, "severity": "Severe" if predicted >= 80 else "Heavy" if predicted >= 65 else "Moderate" if predicted >= 40 else "Low", "recommendation": action, "model_type": "transparent_baseline_demo", "warning": "Illustrative output based on synthetic data; not for real-world traffic control."}

@app.get("/api/signals")
def signals():
    return {"items": [
        {"intersection": "Avinashi Rd × Lakshmi Mills", "green_seconds": 42, "queue_estimate": 28, "recommendation": "Extend green by 12s", "status": "Optimized"},
        {"intersection": "Trichy Rd × Sungam", "green_seconds": 35, "queue_estimate": 19, "recommendation": "Maintain current timing", "status": "Stable"},
        {"intersection": "Sathy Rd × Ganapathy", "green_seconds": 26, "queue_estimate": 41, "recommendation": "Prioritize northbound", "status": "Needs action"},
    ], "mode": "recommendations_only"}
