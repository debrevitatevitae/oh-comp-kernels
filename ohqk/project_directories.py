from pathlib import Path

PROJECT_ABSOLUTE_PATH = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ABSOLUTE_PATH / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PICKLE_DIR = PROJECT_ABSOLUTE_PATH / "pickle"
PROC_DATA_DIR = DATA_DIR / "processed"
GRAPHICS_DIR = PROJECT_ABSOLUTE_PATH / "graphics"
RESULTS_DIR = PROJECT_ABSOLUTE_PATH / "results"
