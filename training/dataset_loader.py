# import os
# import librosa
# import numpy as np

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATASET_PATH = os.path.join(BASE_DIR, "dataset")

# def extract_features(file_path):
#     try:
#         y, sr = librosa.load(file_path, sr=None)

#         if len(y) < sr * 0.5:
#             return None

#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#         chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#         spectral = librosa.feature.spectral_centroid(y=y, sr=sr)

#         features = np.hstack([
#             np.mean(mfcc.T, axis=0),
#             np.mean(chroma.T, axis=0),
#             np.mean(spectral.T)
#         ])

#         return features

#     except Exception as e:
#         print(f"Skipping {file_path}")
#         return None


# def load_dataset():
#     X, y = [], []

#     classes = {
#         "human": 0,
#         "ai": 1
#     }

#     for class_name, label in classes.items():
#         class_path = os.path.join(DATASET_PATH, class_name)

#         if not os.path.isdir(class_path):
#             continue

#         for file in os.listdir(class_path):
#             if file.lower().endswith((".mp3", ".wav")):
#                 file_path = os.path.join(class_path, file)
#                 features = extract_features(file_path)

#                 if features is not None:
#                     X.append(features)
#                     y.append(label)

#     return np.array(X), np.array(y)


import os
import librosa
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)

        if len(y) < sr * 0.5:
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        return mfcc_mean

    except Exception:
        print(f"Skipping {file_path}")
        return None


def load_dataset():
    X, y = [], []

    classes = {
        "human": 0,
        "ai": 1
    }

    for class_name, label in classes.items():
        class_path = os.path.join(DATASET_PATH, class_name)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            if file.lower().endswith(".mp3"):
                file_path = os.path.join(class_path, file)
                features = extract_features(file_path)

                if features is not None:
                    X.append(features)
                    y.append(label)

    return np.array(X), np.array(y)
