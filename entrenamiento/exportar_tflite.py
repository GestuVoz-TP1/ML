"""Paso 4: exporta los modelos a TensorFlow Lite y mide la latencia en la CPU.

- gestos_fotograma.tflite: imagen 224x224 RGB -> (características, probabilidades)
- gestos_secuencia.tflite: 16 x características -> probabilidades del gesto
La latencia se mide en la CPU del computador; la medición válida para OB2 es
la del teléfono, que se hará al integrar el modelo en la aplicación.
"""
import json
import time

from comun import CRITERIOS, LONG_SECUENCIA, SALIDA, TAM, activar_gpu, clases

activar_gpu()
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


def convertir(funcion, ruta):
    conv = tf.lite.TFLiteConverter.from_concrete_functions([funcion.get_concrete_function()], funcion)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    ruta.write_bytes(conv.convert())
    return ruta


def latencia(ruta, forma, repeticiones=100):
    interp = tf.lite.Interpreter(model_path=str(ruta), num_threads=4)
    interp.allocate_tensors()
    entrada = interp.get_input_details()[0]["index"]
    dato = np.random.rand(*forma).astype("float32")
    for _ in range(10):
        interp.set_tensor(entrada, dato)
        interp.invoke()
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        interp.set_tensor(entrada, dato)
        interp.invoke()
    return (time.perf_counter() - t0) / repeticiones * 1000


def main():
    destino = SALIDA / "tflite"
    destino.mkdir(parents=True, exist_ok=True)
    fotograma = tf.keras.models.load_model(SALIDA / "mobilenet" / "mejor.keras")
    secuencia = tf.keras.models.load_model(SALIDA / "lstm" / "lstm.keras")
    extractor = tf.keras.Model(fotograma.input, [fotograma.get_layer("caracteristicas").output, fotograma.output])
    dim = extractor.outputs[0].shape[-1]

    f1 = tf.function(lambda x: extractor(x, training=False),
                     input_signature=[tf.TensorSpec([1, TAM, TAM, 3], tf.float32, name="imagen")])
    f2 = tf.function(lambda x: secuencia(x, training=False),
                     input_signature=[tf.TensorSpec([1, LONG_SECUENCIA, dim], tf.float32, name="caracteristicas")])
    r1 = convertir(f1, destino / "gestos_fotograma.tflite")
    r2 = convertir(f2, destino / "gestos_secuencia.tflite")
    (destino / "etiquetas.txt").write_text("\n".join(clases()) + "\n", encoding="utf-8")

    ms1 = latencia(r1, (1, TAM, TAM, 3))
    ms2 = latencia(r2, (1, LONG_SECUENCIA, dim))
    # En la app, MobileNet procesa cada fotograma al llegar; al cerrar la ventana solo falta el LSTM.
    info = {
        "fotograma": {"archivo": r1.name, "kb": round(r1.stat().st_size / 1024), "ms_cpu_pc": round(ms1, 2)},
        "secuencia": {"archivo": r2.name, "kb": round(r2.stat().st_size / 1024), "ms_cpu_pc": round(ms2, 2)},
        "ms_ultimo_fotograma_mas_lstm": round(ms1 + ms2, 2),
        "criterio_ms": CRITERIOS["latencia_ms"],
        "nota": "Medido en la CPU del computador de entrenamiento; falta medir en el teléfono.",
    }
    (destino / "latencia.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
