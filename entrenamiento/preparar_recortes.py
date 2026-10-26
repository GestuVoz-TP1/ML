"""Paso 1: recorta las manos de HaGRIDv2 y arma el subconjunto de entrenamiento.

Cada gesto del repertorio toma sus imágenes de la clase HaGRID equivalente.
La clase 'ninguno' mezcla manos sin gesto (etiqueta no_gesture) y gestos de
HaGRID que no están en el repertorio, para que el modelo aprenda a rechazarlos.
"""
import argparse
import csv
import json
import random
from multiprocessing import Pool

from PIL import Image

from comun import ANOTACIONES, IMAGENES, NINGUNO, RECORTES, TAM, cargar_repertorio

POR_SPLIT = {"train": 3000, "val": 400, "test": 1000}
MARGEN = 1.6  # el recorte cubre 1.6 veces la mano, para incluir algo de contexto (p. ej., la boca en Silencio)


def caja_del_gesto(ann, etiqueta):
    """Caja normalizada [x, y, w, h] del gesto; en gestos de dos manos, la caja que las une."""
    if ann.get("united_bbox") and ann.get("united_label") == etiqueta:
        return ann["united_bbox"]
    cajas = [b for b, l in zip(ann["bboxes"], ann["labels"]) if l == etiqueta]
    if not cajas:
        return None
    x0 = min(b[0] for b in cajas)
    y0 = min(b[1] for b in cajas)
    x1 = max(b[0] + b[2] for b in cajas)
    y1 = max(b[1] + b[3] for b in cajas)
    return [x0, y0, x1 - x0, y1 - y0]


def recortar(tarea):
    origen, caja, destino = tarea
    try:
        img = Image.open(origen).convert("RGB")
    except (OSError, FileNotFoundError):
        return None
    w, h = img.size
    x, y, bw, bh = caja[0] * w, caja[1] * h, caja[2] * w, caja[3] * h
    lado = max(bw, bh) * MARGEN
    cx, cy = x + bw / 2, y + bh / 2
    recorte = img.crop((int(cx - lado / 2), int(cy - lado / 2), int(cx + lado / 2), int(cy + lado / 2)))
    recorte.resize((TAM, TAM), Image.BILINEAR).save(destino, quality=92)
    return destino


def tareas_de_split(split, repertorio, rng):
    jsons = sorted(p.stem for p in (ANOTACIONES / split).glob("*.json"))
    cupo = POR_SPLIT[split]
    gestos, ninguno = {}, []
    for clase_hagrid in jsons:
        with open(ANOTACIONES / split / f"{clase_hagrid}.json", encoding="utf-8") as f:
            anns = json.load(f)
        ids = list(anns)
        rng.shuffle(ids)
        carpeta = IMAGENES / clase_hagrid
        if clase_hagrid in repertorio:
            lista = []
            for image_id in ids:
                caja = caja_del_gesto(anns[image_id], clase_hagrid)
                if caja:
                    lista.append((carpeta / f"{image_id}.jpg", caja, clase_hagrid))
                if len(lista) >= cupo:
                    break
            gestos[repertorio[clase_hagrid]] = lista
        # Negativos: manos sin gesto de cualquier imagen y gestos que no usamos
        for image_id in ids[:400]:
            ann = anns[image_id]
            for b, l in zip(ann["bboxes"], ann["labels"]):
                if l == "no_gesture" or (l not in repertorio and l == clase_hagrid):
                    ninguno.append((carpeta / f"{image_id}.jpg", b, l))
    rng.shuffle(ninguno)
    gestos[NINGUNO] = ninguno[: cupo * 2]
    return gestos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--procesos", type=int, default=6)
    args = parser.parse_args()

    repertorio = cargar_repertorio()
    rng = random.Random(42)
    indice = []
    for split in POR_SPLIT:
        print(f"[{split}] leyendo anotaciones...")
        gestos = tareas_de_split(split, repertorio, rng)
        tareas = []
        for clase, lista in gestos.items():
            carpeta = RECORTES / split / clase
            carpeta.mkdir(parents=True, exist_ok=True)
            for i, (origen, caja, etiqueta) in enumerate(lista):
                destino = carpeta / f"{clase}_{i:05d}.jpg"
                tareas.append((str(origen), caja, str(destino)))
                indice.append((split, clase, etiqueta, origen.name, destino.name))
            print(f"  {clase:16s} {len(lista):5d}")
        with Pool(args.procesos) as pool:
            hechos = sum(1 for r in pool.imap_unordered(recortar, tareas, chunksize=64) if r)
        print(f"  recortes guardados: {hechos}/{len(tareas)}")

    with open(RECORTES / "indice.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("split", "clase", "etiqueta_hagrid", "imagen", "recorte"), *indice])


if __name__ == "__main__":
    main()
