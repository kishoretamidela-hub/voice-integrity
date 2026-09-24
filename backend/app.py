import os
import sys
import io
import time
import random
import subprocess
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import librosa
import numpy as np
import imageio_ffmpeg
import requests


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "voice_cnn.pth"

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

DATASET_REAL_DIR = (
    BASE_DIR / "dataset" / "for-2seconds" / "testing" / "real"
)

DATASET_FAKE_DIR = (
    BASE_DIR / "dataset" / "for-2seconds" / "testing" / "fake"
)


# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path=""
)

CORS(app)


# ============================================================================
# MODEL DEFINITION & LOADING
# ============================================================================

class VoiceCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------------------
            # Block 1
            # ------------------------------------------------------------

            nn.Conv2d(
                1,
                32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.10),


            # ------------------------------------------------------------
            # Block 2
            # ------------------------------------------------------------

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.15),


            # ------------------------------------------------------------
            # Block 3
            # ------------------------------------------------------------

            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.20),


            # ------------------------------------------------------------
            # Block 4
            # ------------------------------------------------------------

            nn.Conv2d(
                128,
                256,
                3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                1
            )
        )


    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


# ============================================================================
# LOAD MODEL
# ============================================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = VoiceCNN().to(device)


if MODEL_PATH.exists():

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    print(
        f"✓ VoiceCNN model loaded from: {MODEL_PATH}"
    )

else:

    print(
        f"Warning: Model weights not found at {MODEL_PATH}"
    )


model.eval()

print("Device:", device)


# ============================================================================
# IN-MEMORY OTP STORE
# ============================================================================

# Format:
# phone_number -> {
#     "otp": "123456",
#     "expires_at": timestamp,
#     "created_at": timestamp
# }

otp_store = {}


# ============================================================================
# AUDIO CONVERSION & PREPROCESSING
# ============================================================================

def convert_to_wav(audio_bytes):
    """
    Converts browser WebM/Opus/MP3/OGG audio
    into 16kHz mono WAV.
    """

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [

        ffmpeg,

        "-y",

        "-i",
        "pipe:0",

        "-f",
        "wav",

        "-ar",
        "16000",

        "-ac",
        "1",

        "pipe:1"
    ]


    result = subprocess.run(

        command,

        input=audio_bytes,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE
    )


    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg audio conversion failed: "
            + result.stderr.decode(
                "utf-8",
                errors="ignore"
            )
        )


    return result.stdout


def audio_slice_to_mel(audio_slice):
    """
    Converts a 2-second float32 16kHz audio slice
    into a normalized Mel-Spectrogram tensor.
    """

    target_len = 16000 * 2


    if len(audio_slice) < target_len:

        audio_slice = np.pad(
            audio_slice,
            (
                0,
                target_len - len(audio_slice)
            )
        )

    else:

        audio_slice = audio_slice[
            :target_len
        ]


    mel = librosa.feature.melspectrogram(

        y=audio_slice,

        sr=16000,

        n_fft=1024,

        hop_length=256,

        n_mels=128
    )


    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )


    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-8
    )


    tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    )


    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Shape:
    # (1, 1, 128, time_steps)

    return tensor


# ============================================================================
# AURIGIN AI ANALYSIS
# ============================================================================

def analyze_with_aurigin(wav_bytes):
    """
    Sends WAV audio to Aurigin AI voice detection API.
    """

    api_key = os.getenv(
        "AURIGIN_API_KEY"
    )


    if not api_key:

        raise RuntimeError(
            "AURIGIN_API_KEY environment variable is not set."
        )


    url = "https://api.aurigin.ai/v1/predict"


    headers = {
        "x-api-key": api_key
    }


    files = {

        "file": (
            "call.wav",
            wav_bytes,
            "audio/wav"
        )
    }


    response = requests.post(

        url,

        headers=headers,

        files=files,

        timeout=60
    )


    if response.status_code != 200:

        raise RuntimeError(

            f"Aurigin API error "
            f"{response.status_code}: "
            f"{response.text}"
        )


    return response.json()


# ============================================================================
# STATIC ROUTES
# ============================================================================

@app.route("/")
def index():

    return send_from_directory(
        str(STATIC_DIR),
        "index.html"
    )


# ============================================================================
# VOICE PREDICTION ENGINE
# CNN + AURIGIN
# ============================================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    """
    Receives call audio and analyzes it using:

    1. Local VoiceCNN
    2. Aurigin AI

    Returns both results and a final risk assessment.
    """


    # ------------------------------------------------------------------------
    # CHECK AUDIO
    # ------------------------------------------------------------------------

    if "audio" not in request.files:

        return jsonify({

            "error":
            "No audio file stream received"

        }), 400


    audio_file = request.files["audio"]

    audio_bytes = audio_file.read()


    if len(audio_bytes) == 0:

        return jsonify({

            "error":
            "Empty audio stream"

        }), 400


    try:

        # ====================================================================
        # STEP 1: CONVERT AUDIO
        # ====================================================================

        wav_bytes = convert_to_wav(
            audio_bytes
        )


        # ====================================================================
        # STEP 2: LOCAL CNN ANALYSIS
        # ====================================================================

        audio, sr = librosa.load(

            io.BytesIO(wav_bytes),

            sr=16000,

            duration=20
        )


        total_duration = float(
            len(audio) / sr
        )


        # --------------------------------------------------------------------
        # 2-second windows
        # --------------------------------------------------------------------

        window_size = 16000 * 2

        # 1-second hop

        hop_size = 16000 * 1


        slices = []


        if len(audio) <= window_size:

            slices.append(audio)

        else:

            for start in range(

                0,

                len(audio) - window_size + 1,

                hop_size
            ):

                slices.append(

                    audio[
                        start:
                        start + window_size
                    ]
                )


            # Include final partial window

            if len(audio) % hop_size != 0:

                slices.append(
                    audio[-window_size:]
                )


        # --------------------------------------------------------------------
        # CNN predictions
        # --------------------------------------------------------------------

        window_probs = []


        with torch.no_grad():

            mel_batch = torch.cat(
                [
                    audio_slice_to_mel(s)
                    for s in slices
                ],
                dim=0
            ).to(device)


            logits = model(
                mel_batch
            )


            probs = torch.sigmoid(
                logits
            ).flatten().tolist()


            window_probs = [
                round(p, 4)
                for p in probs
            ]


        # --------------------------------------------------------------------
        # Combine window predictions
        # --------------------------------------------------------------------

        max_prob = (
            max(window_probs)
            if window_probs
            else 0.0
        )


        avg_prob = (

            sum(window_probs)
            /
            len(window_probs)

            if window_probs

            else 0.0
        )


        # Weighted combination:
        #
        # 55% strongest suspicious window
        # 45% overall average

        composite_prob = (

            0.55 * max_prob
            +
            0.45 * avg_prob
        )


        cnn_ai_confidence = round(

            composite_prob * 100,

            1
        )


        cnn_human_confidence = round(

            (1.0 - composite_prob) * 100,

            1
        )


        # ====================================================================
        # CNN THRESHOLD
        # ====================================================================

        CNN_AI_THRESHOLD = 72.0


        cnn_is_ai = (
            cnn_ai_confidence
            >= CNN_AI_THRESHOLD
        )


        # ====================================================================
        # STEP 3: AURIGIN ANALYSIS
        # ====================================================================

        aurigin_result = analyze_with_aurigin(
            wav_bytes
        )


        aurigin_global = (
            aurigin_result.get(
                "global",
                {}
            )
        )


        aurigin_label = (
            aurigin_global.get(
                "result",
                "unknown"
            )
        )


        aurigin_confidence = (
            aurigin_global.get(
                "confidence",
                0.0
            )
        )


        aurigin_score = (
            aurigin_global.get(
                "score",
                0.0
            )
        )


        # --------------------------------------------------------------------
        # Normalize Aurigin result
        # --------------------------------------------------------------------

        aurigin_result_lower = str(
            aurigin_label
        ).strip().lower()


        # Aurigin considers these suspicious

        aurigin_suspicious = (

            aurigin_result_lower
            in [
                "spoofed",
                "partially_spoofed"
            ]
        )


        aurigin_is_ai = (
            aurigin_suspicious
        )


        # ====================================================================
        # STEP 4: FINAL DECISION
        # ====================================================================

        if aurigin_suspicious:

            # ---------------------------------------------------------------
            # Aurigin says spoofed / partially spoofed
            # ---------------------------------------------------------------

            if cnn_is_ai:

                final_result = (
                    "AI Synthetic Voice Detected"
                )

                final_risk = "CRITICAL"

            else:

                final_result = (
                    "Potential AI / Spoofed Voice Detected"
                )

                final_risk = "HIGH"


        elif aurigin_result_lower == "bonafide":

            # ---------------------------------------------------------------
            # Aurigin explicitly says genuine human
            # ---------------------------------------------------------------

            final_result = (
                "Likely Genuine Human Voice"
            )

            final_risk = "SAFE"


        elif cnn_is_ai:

            # ---------------------------------------------------------------
            # Aurigin is unknown / unavailable result,
            # so CNN can act as the fallback detector.
            # ---------------------------------------------------------------

            final_result = (
                "Potential Synthetic Voice Detected"
            )

            final_risk = "HIGH"


        else:

            # ---------------------------------------------------------------
            # Neither detector found strong evidence.
            # ---------------------------------------------------------------

            final_result = (
                "Likely Genuine Human Voice"
            )

            final_risk = "SAFE"


        # ====================================================================
        # STEP 5: LOG RESULTS
        # ====================================================================

        print(
            "\n======================================================="
        )

        print(
            "[VOICE INTEGRITY ANALYSIS]"
        )

        print(
            "======================================================="
        )


        print(
            f"Audio duration       : "
            f"{total_duration:.2f}s"
        )


        # --------------------------------------------------------------------
        # CNN
        # --------------------------------------------------------------------

        print(
            "\nLOCAL CNN"
        )


        print(
            f"AI confidence        : "
            f"{cnn_ai_confidence}%"
        )


        print(
            f"Human confidence     : "
            f"{cnn_human_confidence}%"
        )


        print(
            f"CNN threshold        : "
            f"{CNN_AI_THRESHOLD}%"
        )


        print(
            f"CNN AI detected      : "
            f"{cnn_is_ai}"
        )


        # --------------------------------------------------------------------
        # AURIGIN
        # --------------------------------------------------------------------

        print(
            "\nAURIGIN"
        )


        print(
            f"Result               : "
            f"{aurigin_label}"
        )


        print(
            f"Confidence           : "
            f"{round(aurigin_confidence * 100, 2)}%"
        )


        print(
            f"Score                : "
            f"{round(aurigin_score * 100, 2)}%"
        )


        print(
            f"AI detected          : "
            f"{aurigin_is_ai}"
        )


        # --------------------------------------------------------------------
        # FINAL
        # --------------------------------------------------------------------

        print(
            "\nFINAL"
        )


        print(
            f"Result               : "
            f"{final_result}"
        )


        print(
            f"Risk level           : "
            f"{final_risk}"
        )


        print(
            "=======================================================\n"
        )


        # ====================================================================
        # STEP 6: RETURN RESULTS TO FRONTEND
        # ====================================================================

        return jsonify({

            # ----------------------------------------------------------------
            # Final result
            # ----------------------------------------------------------------

            "result":
                final_result,

            "is_ai":
                final_risk != "SAFE",

            "risk_level":
                final_risk,


            # ----------------------------------------------------------------
            # Local CNN
            # ----------------------------------------------------------------

            "cnn": {

                "is_ai":
                    cnn_is_ai,

                "ai_confidence":
                    cnn_ai_confidence,

                "human_confidence":
                    cnn_human_confidence,

                "threshold":
                    CNN_AI_THRESHOLD,

                "window_scores":
                    window_probs,

                "windows_evaluated":
                    len(slices)
            },


            # ----------------------------------------------------------------
            # Aurigin
            # ----------------------------------------------------------------

            "aurigin": {

                "result":
                    aurigin_label,

                "is_ai":
                    aurigin_is_ai,

                "confidence":
                    round(
                        aurigin_confidence * 100,
                        2
                    ),

                "score":
                    round(
                        aurigin_score * 100,
                        2
                    ),

                "prediction_id":
                    aurigin_result.get(
                        "prediction_id"
                    ),

                "model":
                    aurigin_result.get(
                        "model"
                    ),

                "processing_time":
                    aurigin_result.get(
                        "processing_time"
                    )
            },


            # ----------------------------------------------------------------
            # Audio information
            # ----------------------------------------------------------------

            "duration_seconds":
                round(
                    total_duration,
                    1
                ),


            # ----------------------------------------------------------------
            # Explanation
            # ----------------------------------------------------------------

            "details": (

                "Analysis performed using "
                "both the local VoiceCNN and "
                "Aurigin AI voice detection. "

                f"Final assessment: "
                f"{final_result}."
            )
        })


    except Exception as e:

        print(
            "PREDICTION ERROR:",
            repr(e)
        )


        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":

    print(
        "\n======================================================="
    )

    print(
        "VOICEINTEGRITY FULL-STACK SERVICE STARTED"
    )

    print(
        "======================================================="
    )

    print(
        "API & UI URL : "
        "http://localhost:5000"
    )

    print(
        f"Device       : {device}"
    )

    print(
        f"Model Path   : {MODEL_PATH}"
    )

    print(
        "CNN AI Threshold : 72%"
    )

    print(
        "Aurigin Primary  : ENABLED"
    )

    print(
        "=======================================================\n"
    )


    # Render provides PORT through an environment variable.
    # Locally this defaults to 5000.

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False
    )
