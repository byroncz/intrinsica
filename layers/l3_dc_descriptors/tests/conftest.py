import sys
from pathlib import Path

# El lago sintético del lector (`frames_lake`) es el de las pruebas de L3 (TRD-L3 §13).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared/dc_frames/tests"))
