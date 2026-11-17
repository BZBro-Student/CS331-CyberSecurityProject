import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA = os.getenv("DATASETURL")
MODELDIR = Path("models")

MODELPATH = {}

MODELWEIGHT = {}