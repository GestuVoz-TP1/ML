import json
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
HAGRID = Path(os.environ.get("GV_HAGRID", "D:/DATASET/hagridv2_512"))
IMAGENES = HAGRID / "HaGRIDv2_dataset_512"
ANOTACIONES = HAGRID / "annotations"
RECORTES = Path(os.environ.get("GV_RECORTES", "D:/DATASET/gv_recortes"))
SALIDA = Path(os.environ.get("GV_SALIDA", RAIZ / "runs"))

TAM = 224
LONG_SECUENCIA = 16
NINGUNO = "ninguno"

CRITERIOS = {"exactitud": 0.90, "precision": 0.90, "sensibilidad": 0.85, "f1": 0.85, "latencia_ms": 200}


def cargar_repertorio():
    with open(RAIZ / "config" / "gestos.json", encoding="utf-8") as f:
        config = json.load(f)
    return {g["hagrid"]: g["clase"] for g in config["gestos"] if g["hagrid"]}


def clases():
    return list(cargar_repertorio().values()) + [NINGUNO]


def activar_gpu():
    import glob
    import sys

    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for d in glob.glob(str(base / "*" / "bin")):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
