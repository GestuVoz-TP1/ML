import argparse
import json
import re
from pathlib import Path

from comun import NINGUNO, PALMA, PASO_SEGUNDOS, SALIDA, SECUENCIAS, TAM, VIDEOS, activar_gpu, clases

activar_gpu()
import cv2
import numpy as np
import tensorflow as tf

PATRON = re.compile(r"^(P\d\d)_(.+)_c(\d\d)_([a-z_]+)\.mp4$")
ENTRADA_PALMA = 192


class DetectorPalmas:
    def __init__(self, ruta, minimo=0.3, maximo=2):
        self.interp = tf.lite.Interpreter(model_path=str(ruta), num_threads=4)
        self.interp.allocate_tensors()
        self.entrada = self.interp.get_input_details()[0]["index"]
        salidas = self.interp.get_output_details()
        i_reg = 0 if salidas[0]["shape"][-1] == 18 else 1
        self.reg, self.puntajes = salidas[i_reg]["index"], salidas[1 - i_reg]["index"]
        self.minimo, self.maximo = minimo, maximo
        anclas = []
        for paso, por_celda in ((8, 2), (16, 6)):
            g = ENTRADA_PALMA // paso
            for y in range(g):
                for x in range(g):
                    anclas.extend([((x + 0.5) / g, (y + 0.5) / g)] * por_celda)
        self.anclas = np.array(anclas, dtype=np.float32)

    def detectar(self, rgb):
        h, w = rgb.shape[:2]
        lado = max(w, h)
        x = recorte_cuadrado(rgb, w / 2, h / 2, lado, ENTRADA_PALMA) / 255.0
        self.interp.set_tensor(self.entrada, x[None].astype(np.float32))
        self.interp.invoke()
        reg = self.interp.get_tensor(self.reg)[0]
        s = 1 / (1 + np.exp(-np.clip(self.interp.get_tensor(self.puntajes)[0].reshape(-1), -100, 100)))
        ox, oy = (lado - w) / 2, (lado - h) / 2
        palmas, cajas = [], []
        for i in np.argsort(-s):
            if s[i] < self.minimo:
                break
            ax, ay = self.anclas[i]
            cx, cy = reg[i, 0] / ENTRADA_PALMA + ax, reg[i, 1] / ENTRADA_PALMA + ay
            bw, bh = reg[i, 2] / ENTRADA_PALMA, reg[i, 3] / ENTRADA_PALMA
            caja = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            if any(iou(b, caja) > 0.3 for b in cajas):
                continue
            cajas.append(caja)

            def px(v):
                return v * lado - ox

            def py(v):
                return v * lado - oy

            palmas.append(dict(
                cx=px(cx), cy=py(cy), lado=max(bw, bh) * lado,
                muneca=(px(reg[i, 4] / ENTRADA_PALMA + ax), py(reg[i, 5] / ENTRADA_PALMA + ay)),
                medio=(px(reg[i, 8] / ENTRADA_PALMA + ax), py(reg[i, 9] / ENTRADA_PALMA + ay))))
            if len(palmas) == self.maximo:
                break
        return palmas


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def caja_mano(p, escala=3.0, desplazamiento=0.5):
    dx, dy = p["medio"][0] - p["muneca"][0], p["medio"][1] - p["muneca"][1]
    n = (dx * dx + dy * dy) ** 0.5
    ux, uy = (dx / n, dy / n) if n > 0 else (0.0, -1.0)
    return (p["cx"] + ux * desplazamiento * p["lado"], p["cy"] + uy * desplazamiento * p["lado"], p["lado"] * escala)


def caja_union(a, b, margen=1.1):
    izq = min(a[0] - a[2] / 2, b[0] - b[2] / 2)
    arr = min(a[1] - a[2] / 2, b[1] - b[2] / 2)
    der = max(a[0] + a[2] / 2, b[0] + b[2] / 2)
    aba = max(a[1] + a[2] / 2, b[1] + b[2] / 2)
    return ((izq + der) / 2, (arr + aba) / 2, max(der - izq, aba - arr) * margen)


def recorte_cuadrado(rgb, cx, cy, lado, tam):
    escala = tam / lado
    m = np.array([[escala, 0, tam / 2 - cx * escala], [0, escala, tam / 2 - cy * escala]], dtype=np.float32)
    return cv2.warpAffine(rgb, m, (tam, tam), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)).astype(np.float32)


def fotogramas(ruta):
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cuadros, siguiente, i = [], 0.0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / fps
        if t + 1e-6 >= siguiente:
            cuadros.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            siguiente += PASO_SEGUNDOS
        i += 1
    cap.release()
    return cuadros


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default=str(VIDEOS))
    parser.add_argument("--salida", default=str(SECUENCIAS))
    args = parser.parse_args()

    clases_fotograma = clases()
    i_ninguno = clases_fotograma.index(NINGUNO)
    modelo = tf.keras.models.load_model(SALIDA / "mobilenet" / "mejor.keras")
    extractor = tf.keras.Model(modelo.input, [modelo.get_layer("caracteristicas").output, modelo.output])
    detector = DetectorPalmas(PALMA)
    destino = Path(args.salida)
    destino.mkdir(parents=True, exist_ok=True)

    indice = []
    rutas = sorted(Path(args.videos).rglob("*.mp4"))
    for n, ruta in enumerate(rutas, 1):
        m = PATRON.match(ruta.name)
        if not m:
            continue
        persona, clase, numero, variante = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        nombre = ruta.stem
        previo = destino / f"{nombre}.npz"
        if previo.exists():
            total = len(np.load(previo)["caracteristicas"])
            indice.append(dict(archivo=f"{nombre}.npz", persona=persona, clase=clase, numero=numero,
                               variante=variante, fotogramas=total))
            continue
        feats, probs, geom = [], [], []
        for rgb in fotogramas(ruta):
            h, w = rgb.shape[:2]
            cajas = [caja_mano(p) for p in detector.detectar(rgb)]
            hay_mano = 1.0 if cajas else 0.0
            if len(cajas) == 2:
                cajas.append(caja_union(cajas[0], cajas[1]))
            if not cajas:
                cajas = [(w / 2, h / 2, float(min(w, h)))]
            lote = np.stack([recorte_cuadrado(rgb, cx, cy, lado, TAM) for cx, cy, lado in cajas])
            f, p = extractor(lote, training=False)
            f, p = f.numpy(), p.numpy()
            puntaje = np.delete(p, i_ninguno, axis=1).max(1)
            k = int(puntaje.argmax())
            cx, cy, lado = cajas[k]
            feats.append(f[k])
            probs.append(p[k])
            geom.append([cx / w, cy / h, lado / max(w, h), hay_mano])
        np.savez_compressed(destino / f"{nombre}.npz", caracteristicas=np.array(feats, np.float32),
                            probabilidades=np.array(probs, np.float32), geometria=np.array(geom, np.float32))
        indice.append(dict(archivo=f"{nombre}.npz", persona=persona, clase=clase, numero=numero,
                           variante=variante, fotogramas=len(feats)))
        if n % 50 == 0:
            print(f"{n}/{len(rutas)} videos procesados", flush=True)
    (destino / "indice.json").write_text(json.dumps(dict(clases_fotograma=clases_fotograma, videos=indice),
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"listo: {len(indice)} videos en {destino}")


if __name__ == "__main__":
    main()
