# import joblib
# import os

# MODEL_PATH = os.path.join("model", "classifier.pkl")

# model = joblib.load(MODEL_PATH)

import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "classifier.pkl")

model = joblib.load(MODEL_PATH)
