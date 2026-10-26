"""Paso 3: entrena el LSTM que decide el gesto a partir de 16 fotogramas.

HaGRID solo tiene imágenes, así que las secuencias se arman de forma sintética
con las características (capa 'caracteristicas') de MobileNet:
  - sostenido: los 16 fotogramas son del mismo gesto;
  - transición: el gesto empieza o termina dentro de la ventana;
  - ruido: el gesto con 1 a 3 fotogramas de otra clase intercalados;
  - ninguno: manos sin gesto o gestos sueltos que no se mantienen.
Se compara con la votación por mayoría sobre las predicciones por fotograma.
"""
import json

from comun import LONG_SECUENCIA, NINGUNO, RECORTES, SALIDA, TAM, activar_gpu, clases

activar_gpu()
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402

CLASES = clases()
I_NINGUNO = CLASES.index(NINGUNO)
MINIMO_VOTOS = LONG_SECUENCIA // 2  # el gesto debe ocupar al menos la mitad de la ventana


def extraer(modelo, split):
    ds = tf.keras.utils.image_dataset_from_directory(
        RECORTES / split, class_names=CLASES, image_size=(TAM, TAM),
        batch_size=64, shuffle=False, label_mode="int")
    extractor = tf.keras.Model(modelo.input, [modelo.get_layer("caracteristicas").output, modelo.output])
    feats, probs = extractor.predict(ds, verbose=0)
    y = np.concatenate([lbl.numpy() for _, lbl in ds])
    return feats, probs, y


def secuencias(y, n, rng):
    """Devuelve índices de fotogramas (n, 16) y la etiqueta de cada secuencia."""
    por_clase = [np.flatnonzero(y == c) for c in range(len(CLASES))]
    gestos = [c for c in range(len(CLASES)) if c != I_NINGUNO]
    L = LONG_SECUENCIA
    idx = np.zeros((n, L), dtype=np.int64)
    etiquetas = np.zeros(n, dtype=np.int64)
    for i in range(n):
        tipo = rng.random()
        c = rng.choice(gestos)
        tomar = lambda k, clase: rng.choice(por_clase[clase], k)  # noqa: E731
        if tipo < 0.40:  # sostenido
            fila, etiqueta = tomar(L, c), c
        elif tipo < 0.60:  # transición
            k = rng.integers(2, L - MINIMO_VOTOS + 1)
            partes = [tomar(k, I_NINGUNO), tomar(L - k, c)]
            fila, etiqueta = np.concatenate(partes if rng.random() < 0.5 else partes[::-1]), c
        elif tipo < 0.80:  # ruido
            fila = tomar(L, c)
            for p in rng.choice(L, rng.integers(1, 4), replace=False):
                fila[p] = tomar(1, rng.choice([g for g in range(len(CLASES)) if g != c]))[0]
            etiqueta = c
        else:  # ninguno: sin gesto, o gestos sueltos que no se mantienen
            fila = tomar(L, I_NINGUNO)
            for p in rng.choice(L, rng.integers(0, MINIMO_VOTOS - 2), replace=False):
                fila[p] = tomar(1, rng.choice(gestos))[0]
            etiqueta = I_NINGUNO
        idx[i], etiquetas[i] = fila, etiqueta
    return idx, etiquetas


def votacion(probs_seq):
    votos = probs_seq.argmax(-1)
    pred = []
    for fila in votos:
        cuenta = np.bincount(fila, minlength=len(CLASES))
        cuenta[I_NINGUNO] = 0
        g = cuenta.argmax()
        pred.append(g if cuenta[g] >= MINIMO_VOTOS else I_NINGUNO)
    return np.array(pred)


def resumen(reales, pred):
    r = classification_report(reales, pred, target_names=CLASES, output_dict=True, zero_division=0)
    m = r["macro avg"]
    return {"exactitud": r["accuracy"], "precision_macro": m["precision"], "sensibilidad_macro": m["recall"],
            "f1_macro": m["f1-score"], "precision_minima": min(r[c]["precision"] for c in CLASES),
            "por_clase": {c: r[c] for c in CLASES}}


def main():
    destino = SALIDA / "lstm"
    destino.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    tf.keras.utils.set_random_seed(42)
    modelo = tf.keras.models.load_model(SALIDA / "mobilenet" / "mejor.keras")
    datos = {s: extraer(modelo, s) for s in ("train", "val", "test")}

    def armar(split, n):
        feats, probs, y = datos[split]
        idx, etiquetas = secuencias(y, n, rng)
        x = feats[idx] + rng.normal(0, 0.05, (n, LONG_SECUENCIA, feats.shape[1])).astype("float32")
        return x, probs[idx], etiquetas

    x_tr, _, y_tr = armar("train", 40000)
    x_va, _, y_va = armar("val", 4000)
    x_te, p_te, y_te = armar("test", 8000)

    lstm = tf.keras.Sequential([
        tf.keras.Input((LONG_SECUENCIA, x_tr.shape[-1]), name="caracteristicas"),
        tf.keras.layers.LSTM(64, name="lstm"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASES), activation="softmax", name="gesto"),
    ], name="gestuvoz_secuencia")
    lstm.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
    lstm.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=15, batch_size=128,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                                         restore_best_weights=True)])
    lstm.save(destino / "lstm.keras")

    metricas = {"lstm": resumen(y_te, lstm.predict(x_te, verbose=0).argmax(1)),
                "votacion_mayoria": resumen(y_te, votacion(p_te)),
                "secuencias_test": int(len(y_te))}
    with open(destino / "metricas_secuencia.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    for nombre in ("lstm", "votacion_mayoria"):
        m = metricas[nombre]
        print(f"{nombre:17s} exactitud={m['exactitud']:.3f} P={m['precision_macro']:.3f} "
              f"R={m['sensibilidad_macro']:.3f} F1={m['f1_macro']:.3f} Pmin={m['precision_minima']:.3f}")


if __name__ == "__main__":
    main()
