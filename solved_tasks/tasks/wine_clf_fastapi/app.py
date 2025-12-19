from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Wine Quality Predictor")

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

wine_classes = ['Class 0', 'Class 1', 'Class 2']

class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "alcohol": 13.2,
                "malic_acid": 2.77,
                "ash": 2.51,
                "alcalinity_of_ash": 18.5,
                "magnesium": 96.0,
                "total_phenols": 2.45,
                "flavanoids": 2.53,
                "nonflavanoid_phenols": 0.29,
                "proanthocyanins": 1.54,
                "color_intensity": 5.0,
                "hue": 1.04,
                "od280_od315_of_diluted_wines": 3.47,
                "proline": 920.0
            }
        }
    }

@app.get("/")
def read_root():
    return {
        "message": "Wine Quality Prediction API",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Check API health",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": clf is not None, "scaler_loaded": scaler is not None}

@app.post("/predict")
def predict(features: WineFeatures):
    try:
        input_data = np.array([[
            features.alcohol, features.malic_acid, features.ash,
            features.alcalinity_of_ash, features.magnesium,
            features.total_phenols, features.flavanoids,
            features.nonflavanoid_phenols, features.proanthocyanins,
            features.color_intensity, features.hue,
            features.od280_od315_of_diluted_wines, features.proline
        ]])

        input_scaled = scaler.transform(input_data)

        prediction = clf.predict(input_scaled)
        probabilities = clf.predict_proba(input_scaled)[0]
        pred_index = int(prediction[0])

        return {
            "prediction": wine_classes[pred_index],
            "prediction_index": pred_index,
            "confidence": float(probabilities[pred_index]),
            "all_probabilities": {
                wine_classes[i]: float(p) for i, p in enumerate(probabilities)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))