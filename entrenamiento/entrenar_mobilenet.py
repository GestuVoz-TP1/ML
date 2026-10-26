"""Paso 2: entrena el clasificador por fotograma (MobileNetV3 + capa de características).

Etapa 1: la base de MobileNetV3 (pesos de ImageNet) queda congelada y solo se
entrena la cabeza. Etapa 2: se descongela la parte final de la base y se
ajusta con una tasa de aprendizaje baja. Al final evalúa con el split de prueba
de HaGRID (personas que no aparecen en entrenamiento) contra los criterios de OB2.
"""
import argparse
import json

from comun import CRITERIOS, RECORTES, SALIDA, TAM, activar_gpu, clases

activar_gpu()
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

CLASES = clases()
LOTE = 48


def dataset(split, barajar):
    ds = tf.keras.utils.image_dataset_from_directory(
        RECORTES / split, class_names=CLASES, image_size=(TAM, TAM),
        batch_size=LOTE, shuffle=barajar, seed=42, label_mode="int")
    return ds.prefetch(tf.data.AUTOTUNE)


def construir_modelo():
    aumento = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomContrast(0.25),
    ], name="aumento")
    base = tf.keras.applications.MobileNetV3Large(
        input_shape=(TAM, TAM, 3), include_top=False, weights="imagenet",
        pooling="avg", include_preprocessing=True)
    base.trainable = False
    entrada = tf.keras.Input((TAM, TAM, 3), name="imagen")
    x = aumento(entrada)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    caracteristicas = tf.keras.layers.Dense(128, activation="relu", name="caracteristicas")(x)
    x = tf.keras.layers.Dropout(0.3)(caracteristicas)
    salida = tf.keras.layers.Dense(len(CLASES), activation="softmax", name="clase")(x)
    return tf.keras.Model(entrada, salida, name="gestuvoz_fotograma"), base


def pesos_por_clase(ds):
    conteo = np.zeros(len(CLASES))
    for _, y in ds.unbatch():
        conteo[int(y)] += 1
    return {i: conteo.sum() / (len(CLASES) * c) for i, c in enumerate(conteo)}


def evaluar(modelo, destino):
    ds = dataset("test", barajar=False)
    probs = modelo.predict(ds, verbose=0)
    reales = np.concatenate([y.numpy() for _, y in ds])
    pred = probs.argmax(1)
    reporte = classification_report(reales, pred, target_names=CLASES, output_dict=True, zero_division=0)
    matriz = confusion_matrix(reales, pred).tolist()
    macro = reporte["macro avg"]
    resumen = {
        "exactitud": reporte["accuracy"],
        "precision_macro": macro["precision"],
        "sensibilidad_macro": macro["recall"],
        "f1_macro": macro["f1-score"],
        "precision_minima": min(reporte[c]["precision"] for c in CLASES),
        "muestras_test": int(len(reales)),
    }
    umbral = 0.85
    aceptadas = probs.max(1) >= umbral
    resumen["con_umbral_0_85"] = {
        "proporcion_aceptada": float(aceptadas.mean()),
        "exactitud_aceptadas": float((pred[aceptadas] == reales[aceptadas]).mean()),
    }
    resumen["cumple"] = {
        "exactitud": resumen["exactitud"] >= CRITERIOS["exactitud"],
        "precision_por_gesto": resumen["precision_minima"] >= CRITERIOS["precision"],
        "sensibilidad": resumen["sensibilidad_macro"] >= CRITERIOS["sensibilidad"],
        "f1": resumen["f1_macro"] >= CRITERIOS["f1"],
    }
    with open(destino / "metricas_fotograma.json", "w", encoding="utf-8") as f:
        json.dump({"clases": CLASES, "resumen": resumen, "por_clase": {c: reporte[c] for c in CLASES},
                   "matriz_confusion": matriz}, f, ensure_ascii=False, indent=2)
    graficar_matriz(np.array(matriz), destino / "matriz_confusion.png")
    return resumen, reporte


def graficar_matriz(matriz, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = matriz / matriz.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASES)), CLASES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASES)), CLASES)
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas1", type=int, default=4)
    parser.add_argument("--epocas2", type=int, default=8)
    parser.add_argument("--descongelar", type=int, default=60, help="capas finales de la base que se ajustan")
    args = parser.parse_args()

    destino = SALIDA / "mobilenet"
    destino.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)
    entrenamiento, validacion = dataset("train", True), dataset("val", False)
    pesos = pesos_por_clase(entrenamiento)
    modelo, base = construir_modelo()

    guardar = tf.keras.callbacks.ModelCheckpoint(destino / "mejor.keras", monitor="val_accuracy", save_best_only=True)
    historial = {}

    modelo.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
    h1 = modelo.fit(entrenamiento, validation_data=validacion, epochs=args.epocas1,
                    class_weight=pesos, callbacks=[guardar])
    historial["etapa1"] = h1.history

    base.trainable = True
    for capa in base.layers[:-args.descongelar]:
        capa.trainable = False
    for capa in base.layers:
        if isinstance(capa, tf.keras.layers.BatchNormalization):
            capa.trainable = False
    modelo.compile(tf.keras.optimizers.Adam(1e-4), "sparse_categorical_crossentropy", ["accuracy"])
    parar = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    h2 = modelo.fit(entrenamiento, validation_data=validacion, epochs=args.epocas2,
                    class_weight=pesos, callbacks=[guardar, parar])
    historial["etapa2"] = h2.history

    modelo = tf.keras.models.load_model(destino / "mejor.keras")
    with open(destino / "historial.json", "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=2, default=float)
    resumen, reporte = evaluar(modelo, destino)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    for c in CLASES:
        r = reporte[c]
        print(f"{c:16s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")


if __name__ == "__main__":
    main()
