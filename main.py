import re
from contextlib import asynccontextmanager
import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

model = None
tfidf = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tfidf
    try:
        model = joblib.load("job_detector_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        print("ML Models successfully load ho gaye.")
    except Exception as e:
        print(f"Error loading models: {e}")
    yield


app = FastAPI(title="Fake Job Detector API", lifespan=lifespan)


class JobAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=10)


class JobAnalysisResponse(BaseModel):
    is_fake: bool
    risk_level: str
    confidence_score: float | None
    red_flags: list[str]
    message: str


def run_heuristic_checks(text: str) -> list[str]:
    flags = []
    text_lower = text.lower()

    if re.search(r"\b(telegram|whatsapp|crypto|bitcoin|wire transfer|venmo|zelle)\b", text_lower):
        flags.append("Unconventional payment channels mentioned.")
    if re.search(r"\b(registration fee|training fee|processing fee|refundable fee)\b", text_lower):
        flags.append("Upfront payment requested.")
    if re.search(r"\b(no experience|earn \$[0-9]{3,}|guaranteed income|daily payout)\b", text_lower):
        flags.append("Unrealistic income promise for low/no skills.")

    return flags


@app.post("/predict", response_model=JobAnalysisResponse)
def predict_job(payload: JobAnalysisRequest):
    if model is None or tfidf is None:
        raise HTTPException(
            status_code=status.HTTP_53_SERVICE_UNAVAILABLE,
            detail="Models loaded nahi hain."
        )

    text = payload.text.strip()
    vec_text = tfidf.transform([text])
    
    ml_prediction = int(model.predict(vec_text)[0])
    
    confidence_score = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec_text)[0]
        confidence_score = round(float(probs[ml_prediction]), 2)

    red_flags = run_heuristic_checks(text)

    is_fake = ml_prediction == 1 or len(red_flags) > 0

    if ml_prediction == 1 and len(red_flags) > 0:
        risk_level = "HIGH"
    elif is_fake:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return JobAnalysisResponse(
        is_fake=is_fake,
        risk_level=risk_level,
        confidence_score=confidence_score,
        red_flags=red_flags,
        message="Analysis complete.",
    )