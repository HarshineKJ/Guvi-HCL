# import numpy as np
# from app.model import model

# def predict_voice(features):
#     features = np.array(features).reshape(1, -1)
#     prob = model.predict_proba(features)[0]

#     confidence = float(max(prob))
#     prediction = "AI_GENERATED" if model.predict(features)[0] == 1 else "HUMAN"

#     return prediction, confidence

import numpy as np
from app.model import model

def predict_voice(features):
    features = np.array(features).reshape(1, -1)

    # Predict class
    pred = model.predict(features)[0]

    # Try probability-based confidence
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0]
        confidence = float(prob[pred])
    else:
        confidence = 0.5  # fallback if probabilities unavailable

    prediction = "AI_GENERATED" if pred == 1 else "HUMAN"

    return prediction, round(confidence, 3)

