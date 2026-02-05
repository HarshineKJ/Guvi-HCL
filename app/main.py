# from fastapi import FastAPI, Depends
# from pydantic import BaseModel
# from app.auth import verify_api_key
# from app.audio_utils import extract_features
# from app.predict import predict_voice

# app = FastAPI(title="AI Voice Detection API")

# class AudioRequest(BaseModel):
#     audio_base64: str
#     language: str

# @app.post("/detect")
# def detect_voice(data: AudioRequest, api_key: str = Depends(verify_api_key)):
#     features = extract_features(data.audio_base64)
#     label, confidence = predict_voice(features)

#     return {
#         "result": label,
#         "confidence": round(confidence, 3),
#         "language": data.language
#     }


# from fastapi import FastAPI, Depends, HTTPException
# from pydantic import BaseModel
# from app.auth import verify_api_key
# from app.audio_utils import extract_features
# from app.predict import predict_voice

# app = FastAPI(title="AI Voice Detection API")

# SUPPORTED_LANGUAGES = {"en", "ta", "hi", "ml", "te"}

# class AudioRequest(BaseModel):
#     audio_base64: str
#     language: str

# @app.get("/")
# def root():
#     return {"message": "AI Voice Detection API is running"}

# @app.post("/detect")
# def detect_voice(
#     data: AudioRequest,
#     api_key: str = Depends(verify_api_key)
# ):
#     # Validate language
#     if data.language not in SUPPORTED_LANGUAGES:
#         raise HTTPException(
#             status_code=400,
#             detail="Unsupported language"
#         )

#     try:
#         features = extract_features(data.audio_base64)
#         label, confidence = predict_voice(features)

#         return {
#             "result": label,
#             "confidence": round(float(confidence), 3),
#             "language": data.language
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
import numpy as np
import librosa
from app.model import model
from app.auth import verify_api_key

# app = FastAPI(title="AI Voice Detection API")

# SUPPORTED_LANGUAGES = {"en", "ta", "hi", "ml", "te"}

# # --- Existing Base64 endpoint ---
# # Your AudioRequest class and /detect endpoint stay as-is

# # --- New File Upload endpoint ---
# @app.post("/detect_file")
# async def detect_file(
#     file: UploadFile = File(...),
#     api_key: str = Depends(verify_api_key)
# ):
#     try:
#         # Read file bytes
#         file_bytes = await file.read()

#         # Extract features
#         import io, soundfile as sf
#         audio_file = io.BytesIO(file_bytes)
#         y, sr = sf.read(audio_file)
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#         features = np.mean(mfccs.T, axis=0)

#         # Predict
#         features = np.array(features).reshape(1, -1)
#         pred = model.predict(features)[0]
#         confidence = float(model.predict_proba(features)[0][pred]) if hasattr(model, "predict_proba") else 0.5
#         label = "AI_GENERATED" if pred == 1 else "HUMAN"

#         return {
#             "result": label,
#             "confidence": round(confidence, 3)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# from app.audio_utils import extract_features_file
# from app.predict import predict_voice
# from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
# import numpy as np
# import librosa
# from app.model import model
# from app.auth import verify_api_key

# @app.post("/detect_file")
# async def detect_file(
#     file: UploadFile = File(...),
#     api_key: str = Depends(verify_api_key)
# ):
#     try:
#         # Read uploaded file
#         file_bytes = await file.read()

#         # ✅ Extract features from first 2 minutes
#         features = extract_features_file(file_bytes, 120)

#         # Predict
#         label, confidence = predict_voice(features)

#         return {
#             "result": label,
#             "confidence": round(float(confidence), 3)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# 1️⃣ Imports
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from app.auth import verify_api_key
from app.audio_utils import extract_features_file
from app.predict import predict_voice


# 2️⃣ Create FastAPI app FIRST
app = FastAPI(title="AI Voice Detection API")


# 3️⃣ Root endpoint (optional)
@app.get("/")
def root():
    return {"message": "API Running"}


# 4️⃣ File Upload Endpoint
@app.post("/detect_file")
async def detect_file(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    try:
        file_bytes = await file.read()

        # Extract features from first 2 minutes
        features = extract_features_file(file_bytes, 120)

        # Predict
        label, confidence = predict_voice(features)

        return {
            "result": label,
            "confidence": round(float(confidence), 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


