import json
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
HAGRID = Path(os.environ.get("GV_HAGRID", "D:/DATASET/hagridv2_512"))
IMAGENES = HAGRID / "HaGRIDv2_dataset_512"
ANOTACIONES = HAGRID / "annotations"
RECORTES = Path(os.environ.get("GV_RECORTES", "D:/DATASET/gv_recortes_v2"))
VIDEOS = Path(os.environ.get("GV_VIDEOS", "D:/DATASET/dataset"))
SECUENCIAS = Path(os.environ.get("GV_SECUENCIAS", "D:/DATASET/gv_secuencias"))
PALMA = Path(os.environ.get("GV_PALMA", RAIZ.parent / "Mobile-App" / "assets" / "models" / "palm_detection_lite.tflite"))
SALIDA = Path(os.environ.get("GV_SALIDA", RAIZ / "runs"))

TAM = 224
LONG_SECUENCIA = 16
PASO_SEGUNDOS = 0.15
GEOMETRIA = 4
NINGUNO = "ninguno"

CRITERIOS = {"exactitud": 0.90, "precision": 0.90, "sensibilidad": 0.85, "f1": 0.85, "latencia_ms": 200}


def cargar_repertorio():
    with open(RAIZ / "config" / "gestos.json", encoding="utf-8") as f:
        config = json.load(f)
    return {g["hagrid"]: g["clase"] for g in config["gestos"] if g["hagrid"]}


def clases():
    return list(cargar_repertorio().values()) + [NINGUNO]


def clases_secuencia():
    with open(RAIZ / "config" / "gestos.json", encoding="utf-8") as f:
        config = json.load(f)
    return [g["clase"] for g in config["gestos"]] + [NINGUNO]


def activar_gpu():
    import glob
    import sys

    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for d in glob.glob(str(base / "*" / "bin")):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
