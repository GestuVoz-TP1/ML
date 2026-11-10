import argparse
import json

from comun import (LONG_SECUENCIA, NINGUNO, RECORTES, SALIDA, SECUENCIAS, TAM, activar_gpu, clases,
                   clases_secuencia)

activar_gpu()
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

CLASES = clases_secuencia()
I_NINGUNO = CLASES.index(NINGUNO)
SIN_MANO = np.array([0.5, 0.5, 0.75, 0.0], dtype=np.float32)
UMBRAL = 0.85


def extraer_recortes(modelo, split):
    ds = tf.keras.utils.image_dataset_from_directory(
        RECORTES / split, class_names=clases(), image_size=(TAM, TAM),
        batch_size=64, shuffle=False, label_mode="int")
    extractor = tf.keras.Model(modelo.input, modelo.get_layer("caracteristicas").output)
    feats = extractor.predict(ds, verbose=0)
    y = np.concatenate([lbl.numpy() for _, lbl in ds])
    mapa = np.array([CLASES.index(c) for c in clases()])
    return feats, mapa[y]


def geometria_sintetica(rng, n):
    base = np.stack([rng.uniform(0.3, 0.7, n), rng.uniform(0.35, 0.75, n), rng.uniform(0.25, 0.55, n)], 1)
    g = base[:, None, :] + rng.normal(0, 0.02, (n, LONG_SECUENCIA, 3))
    return np.concatenate([g, np.ones((n, LONG_SECUENCIA, 1))], -1).astype(np.float32)


def secuencias_sinteticas(feats, y, n, rng):
    presentes = sorted(set(y.tolist()))
    por_clase = {c: np.flatnonzero(y == c) for c in presentes}
    gestos = [c for c in presentes if c != I_NINGUNO]
    L = LONG_SECUENCIA
    idx = np.zeros((n, L), dtype=np.int64)
    etiquetas = np.zeros(n, dtype=np.int64)
    con_mano = np.ones((n, L), dtype=bool)
    for i in range(n):
        tipo = rng.random()
        c = rng.choice(gestos)
        if tipo < 0.45:
            fila, etiqueta = rng.choice(por_clase[c], L), c
        elif tipo < 0.65:
            k = int(rng.integers(2, L // 2))
            partes = [rng.choice(por_clase[I_NINGUNO], k), rng.choice(por_clase[c], L - k)]
            fila, etiqueta = np.concatenate(partes if rng.random() < 0.5 else partes[::-1]), c
        else:
            fila, etiqueta = rng.choice(por_clase[I_NINGUNO], L), I_NINGUNO
            con_mano[i] = rng.random(L) < 0.6
        idx[i], etiquetas[i] = fila, etiqueta
    g = geometria_sintetica(rng, n)
    g[~con_mano] = SIN_MANO
    return np.concatenate([feats[idx], g], -1), etiquetas


def ventanas_de_videos(indice, raiz):
    xs, ys, personas, videos = [], [], [], []
    for v in indice["videos"]:
        d = np.load(raiz / v["archivo"])
        x = np.concatenate([d["caracteristicas"], d["geometria"]], -1)
        if len(x) < LONG_SECUENCIA:
            continue
        for inicio in range(len(x) - LONG_SECUENCIA + 1):
            xs.append(x[inicio:inicio + LONG_SECUENCIA])
            ys.append(CLASES.index(v["clase"]))
            personas.append(v["persona"])
            videos.append(v["archivo"])
    return np.array(xs, np.float32), np.array(ys), np.array(personas), np.array(videos)


def construir(dim):
    modelo = tf.keras.Sequential([
        tf.keras.Input((LONG_SECUENCIA, dim), name="caracteristicas"),
        tf.keras.layers.GaussianNoise(0.05),
        tf.keras.layers.LSTM(64, name="lstm"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASES), activation="softmax", name="gesto"),
    ], name="gestuvoz_secuencia")
    modelo.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
    return modelo


def entrenar(x, y, pesos, rng, epocas):
    orden = rng.permutation(len(x))
    corte = int(len(x) * 0.9)
    tr, va = orden[:corte], orden[corte:]
    modelo = construir(x.shape[-1])
    modelo.fit(x[tr], y[tr], sample_weight=pesos[tr], validation_data=(x[va], y[va]), epochs=epocas,
               batch_size=128, verbose=2,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                                           restore_best_weights=True)])
    return modelo


def pesos_muestra(y, es_video, factor_video):
    conteo = np.bincount(y, minlength=len(CLASES)).astype(float)
    w = len(y) / (len(CLASES) * np.maximum(conteo, 1))
    return (w[y] * np.where(es_video, factor_video, 1.0)).astype(np.float32)


def resumen(reales, pred):
    etiquetas = list(range(len(CLASES)))
    r = classification_report(reales, pred, labels=etiquetas, target_names=CLASES, output_dict=True,
                              zero_division=0)
    m = r["macro avg"]
    return {"exactitud": float((reales == pred).mean()),
            "precision_macro": m["precision"], "sensibilidad_macro": m["recall"], "f1_macro": m["f1-score"],
            "precision_minima": min(r[c]["precision"] for c in CLASES),
            "por_clase": {c: r[c] for c in CLASES},
            "matriz_confusion": confusion_matrix(reales, pred, labels=etiquetas).tolist()}


def decision_por_video(probs, ys, videos):
    reales, preds, aceptados = [], [], []
    for v in np.unique(videos):
        k = videos == v
        p = probs[k]
        etiquetas = p.argmax(1)
        estable = [etiquetas[i] for i in range(1, len(etiquetas))
                   if etiquetas[i] == etiquetas[i - 1] and etiquetas[i] != I_NINGUNO
                   and p[i].max() >= UMBRAL and p[i - 1].max() >= UMBRAL]
        reales.append(ys[k][0])
        preds.append(estable[0] if estable else I_NINGUNO)
        aceptados.append(bool(estable))
    return np.array(reales), np.array(preds), np.array(aceptados)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sinteticas", type=int, default=20000)
    parser.add_argument("--factor-video", type=float, default=3.0)
    parser.add_argument("--epocas", type=int, default=30)
    args = parser.parse_args()

    destino = SALIDA / "lstm"
    destino.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    tf.keras.utils.set_random_seed(42)

    fotograma = tf.keras.models.load_model(SALIDA / "mobilenet" / "mejor.keras")
    f_tr, y_tr = extraer_recortes(fotograma, "train")
    x_sint, y_sint = secuencias_sinteticas(f_tr, y_tr, args.sinteticas, rng)

    indice = json.loads((SECUENCIAS / "indice.json").read_text(encoding="utf-8"))
    x_vid, y_vid, p_vid, v_vid = ventanas_de_videos(indice, SECUENCIAS)
    personas = sorted(set(p_vid.tolist()))
    print(f"ventanas de video: {len(x_vid)} de {len(set(v_vid.tolist()))} videos; personas: {personas}")

    probs_lopo = np.zeros((len(x_vid), len(CLASES)), np.float32)
    por_persona = {}
    for persona in personas:
        prueba = p_vid == persona
        x = np.concatenate([x_sint, x_vid[~prueba]])
        y = np.concatenate([y_sint, y_vid[~prueba]])
        es_video = np.concatenate([np.zeros(len(x_sint), bool), np.ones((~prueba).sum(), bool)])
        modelo = entrenar(x, y, pesos_muestra(y, es_video, args.factor_video), rng, args.epocas)
        probs_lopo[prueba] = modelo.predict(x_vid[prueba], verbose=0)
        reales, preds, _ = decision_por_video(probs_lopo[prueba], y_vid[prueba], v_vid[prueba])
        por_persona[persona] = {
            "ventanas_exactitud": float((probs_lopo[prueba].argmax(1) == y_vid[prueba]).mean()),
            "videos_exactitud": float((reales == preds).mean()), "videos": int(len(reales))}
        print(persona, por_persona[persona], flush=True)

    np.savez_compressed(destino / "probs_lopo.npz", probs=probs_lopo, y=y_vid, v=v_vid, p=p_vid)
    reales, preds, aceptados = decision_por_video(probs_lopo, y_vid, v_vid)
    metricas = {
        "evaluacion": "dejando fuera a una persona cada vez: entrena con las demas y prueba con ella",
        "personas": personas,
        "clases": CLASES,
        "por_persona": por_persona,
        "ventanas": resumen(y_vid, probs_lopo.argmax(1)),
        "videos_como_en_la_app": resumen(reales, preds),
        "videos_reconocidos_con_umbral": float(aceptados.mean()),
        "umbral": UMBRAL,
    }

    x = np.concatenate([x_sint, x_vid])
    y = np.concatenate([y_sint, y_vid])
    es_video = np.concatenate([np.zeros(len(x_sint), bool), np.ones(len(x_vid), bool)])
    final = entrenar(x, y, pesos_muestra(y, es_video, args.factor_video), rng, args.epocas)
    final.save(destino / "lstm.keras")
    with open(destino / "metricas_secuencia.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    for nombre in ("ventanas", "videos_como_en_la_app"):
        m = metricas[nombre]
        print(f"{nombre:22s} exactitud={m['exactitud']:.3f} P={m['precision_macro']:.3f} "
              f"R={m['sensibilidad_macro']:.3f} F1={m['f1_macro']:.3f} Pmin={m['precision_minima']:.3f}")
    for c in CLASES:
        r = metricas["videos_como_en_la_app"]["por_clase"][c]
        print(f"  {c:16s} P={r['precision']:.2f} R={r['recall']:.2f} n={int(r['support'])}")


if __name__ == "__main__":
    main()
