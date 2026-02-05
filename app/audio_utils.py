# import base64
# import librosa
# import numpy as np
# import tempfile

# def extract_features(base64_audio: str):
#     audio_bytes = base64.b64decode(base64_audio)

#     with tempfile.NamedTemporaryFile(suffix=".mp3") as temp:
#         temp.write(audio_bytes)
#         temp.flush()
#         y, sr = librosa.load(temp.name, sr=None)

#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     mfcc_mean = np.mean(mfcc.T, axis=0)

#     return mfcc_mean

# import base64
# import librosa
# import numpy as np
# import tempfile

# def extract_features(base64_audio: str):
#     try:
#         # Decode Base64 → bytes
#         audio_bytes = base64.b64decode(base64_audio)

#         # Write bytes to a temporary MP3 file
#         with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp:
#             temp.write(audio_bytes)
#             temp.flush()

#             # Load MP3 audio
#             y, sr = librosa.load(temp.name, sr=16000)

#         # Safety check
#         if y is None or len(y) == 0:
#             raise ValueError("Empty or invalid audio")

#         # Extract MFCC features
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

#         # Take mean over time axis
#         mfcc_mean = np.mean(mfcc.T, axis=0)

#         return mfcc_mean

#     except Exception as e:
#         raise ValueError(f"MP3 audio processing failed: {str(e)}")
# import base64
# import librosa
# import numpy as np
# import tempfile
# import os

# def extract_features(base64_audio: str):
#     try:
#         # Decode base64
#         audio_bytes = base64.b64decode(base64_audio)

#         # Create temp mp3 file (Windows-safe)
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
#             temp.write(audio_bytes)
#             temp_path = temp.name

#         # Load audio AFTER closing file
#         y, sr = librosa.load(temp_path, sr=None)

#         # Clean up temp file
#         os.remove(temp_path)

#         # Feature extraction
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#         mfcc_mean = np.mean(mfcc.T, axis=0)

#         return mfcc_mean

#     except Exception as e:
#         raise RuntimeError(f"MP3 audio processing failed: {str(e)}")


import base64
import librosa
import numpy as np
import tempfile
import os
import soundfile as sf


# ==============================
# Base64 Feature Extraction
# ==============================
def extract_features(base64_audio: str):
    try:
        audio_bytes = base64.b64decode(base64_audio)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
            temp.write(audio_bytes)
            temp_path = temp.name

        y, sr = librosa.load(temp_path, sr=None)
        os.remove(temp_path)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        return mfcc_mean

    except Exception as e:
        raise RuntimeError(f"MP3 audio processing failed: {str(e)}")


# ==============================
# FILE → 2 MINUTE CLIP FEATURE
# ==============================
def extract_features_file(file_bytes, duration_sec=120):
    """
    Extract MFCC from first 2 minutes only
    """

    try:
        # Load audio from bytes
        import io
        audio_file = io.BytesIO(file_bytes)
        y, sr = sf.read(audio_file)

        # Convert stereo → mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Clip to first 2 minutes
        max_samples = duration_sec * sr
        y = y[:max_samples]

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        return mfcc_mean

    except Exception as e:
        raise RuntimeError(f"File audio processing failed: {str(e)}")
