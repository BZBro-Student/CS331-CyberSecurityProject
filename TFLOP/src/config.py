import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA = os.getenv("DATASETURL")
MODELDIR = Path("models")
MODELDIR.mkdir(parents=True, exist_ok=True)

MODELPATH = {
    "sender": MODELDIR / "sender_model.keras",
    "subject": MODELDIR / "subject_model.keras",
    "body": MODELDIR / "body_model.keras",
    "urls": MODELDIR / "urls_model.keras",
    "ensemble": MODELDIR / "phishing_ensemble.keras",
}

MODELWEIGHT = {
    "sender": 0.25,
    "subject": 0.25,
    "body": 0.40,
    "urls": 0.10,
}