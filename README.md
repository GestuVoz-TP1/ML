# GestuVoz — Modelo de reconocimiento de gestos

Entorno de entrenamiento del componente de Inteligencia Artificial de
**GestuVoz**, sistema aumentativo y alternativo de comunicación (SAAC) para
apoyar la comunicación no verbal de niños con TEA en Lima Metropolitana.

El modelo reconoce los gestos que el niño hace frente a la cámara del teléfono
y la aplicación los convierte en un pictograma y un mensaje de voz. La
inferencia se ejecuta en el dispositivo Android con TensorFlow Lite: las
imágenes nunca salen del teléfono.

| Repositorio | Contenido |
|-------------|-----------|
| [Mobile-App](https://github.com/GestuVoz-TP1/Mobile-App) | Aplicación Flutter, donde se integra el modelo `.tflite` |
| [Backend](https://github.com/GestuVoz-TP1/Backend) | API Django y Cloud Firestore |
| **ML** (este) | Definición de gestos, entrenamiento y exportación del modelo |

---

## Modelo

```
16 fotogramas (≈ 3 s, cámara frontal, color)
   └─► MobileNetV3 (preentrenado con HaGRID), por fotograma
         └─► LSTM (movimiento a lo largo de la secuencia)
               └─► 14 clases: 13 gestos + "ninguno"
```

- **MobileNet** extrae las características de cada fotograma; se parte de
  MobileNetV3-Large preentrenado en ImageNet y se ajusta con las imágenes de
  [HaGRIDv2](https://github.com/hukenovs/hagrid).
- **LSTM** analiza el movimiento y distingue, por ejemplo, Hola (mano que se
  mueve) de Parar (mano quieta).
- La clase **ninguno** permite responder "No reconocido" en lugar de
  comunicar un mensaje incorrecto.
- Salida: dos modelos `.tflite` (float16) embebidos en la aplicación:
  `gestos_fotograma.tflite` (MobileNet) y `gestos_secuencia.tflite` (LSTM).

## Repertorio de gestos

Definido en [`config/gestos.json`](config/gestos.json), que es la fuente
única de las clases, sus etiquetas y su relación con los pictogramas de la
aplicación. **Pendiente de validación por el asesor especializado.**

La justificación (investigación sobre niños con TEA, uso de HaGRID, pares con
riesgo de confusión) y el protocolo completo de recolección de datos están en
[`docs/repertorio-gestos.html`](docs/repertorio-gestos.html).

| Prioridad | Clase | Gesto | HaGRID | Mensaje de voz |
|----------:|-------|-------|--------|----------------|
| 1 | `gesto_ayuda` | Ayuda | `one` | Necesito ayuda |
| 2 | `gesto_parar` | Parar | `stop` | Para, por favor |
| 3 | `gesto_no` | No | `dislike` | No |
| 4 | `gesto_si` | Sí | `like` | Sí |
| 5 | `gesto_dolor` | Me duele | propio | Me duele algo |
| 6 | `gesto_descanso` | Descanso | `timeout` | Necesito un descanso |
| 7 | `gesto_beber` | Beber | propio | Quiero agua |
| 8 | `gesto_comer` | Comer | propio | Quiero comer |
| 9 | `gesto_silencio` | Silencio | `mute` | Quiero silencio |
| 10 | `gesto_porfavor` | Por favor | `holy` | Por favor |
| 11 | `gesto_termine` | Terminé | `xsign` | Ya terminé |
| 12 | `gesto_hola` | Hola | `palm` + movimiento | Hola |
| 13 | `gesto_abrazo` | Abrazo | `hand_heart` | Quiero un abrazo |
| — | `ninguno` | Clase de rechazo | `no_gesture` | "No reconocido" |

Si el modelo no alcanza los criterios de aceptación con los 13 gestos, se
retiran primero los de menor prioridad.

## Conjunto de datos

- **Propio:** clips de 16 fotogramas grabados con la herramienta de grabación
  de la aplicación (compilación de desarrollo), por **adultos voluntarios**
  con consentimiento informado. No se graba a niños en esta fase.
- **Referencia pública:** HaGRIDv2 (adultos, imágenes estáticas). Se usan sus
  modelos preentrenados y, si hace falta, muestras de las clases del
  repertorio.
- Los datos se guardan en Google Drive con acceso restringido al equipo. **No
  se suben a este repositorio** (ver `.gitignore`).

Estructura esperada en Drive:

```
dataset/
├── gesto_ayuda/
│   ├── P01_0001/      16 imágenes (00.jpg … 15.jpg)
│   └── P01_0002/
├── gesto_parar/
├── …
└── ninguno/
```

La división en entrenamiento, validación y prueba se hace **por
participante**, para que las métricas no salgan infladas.

## Entrenamiento

Scripts en [`entrenamiento/`](entrenamiento/), en orden:

| Paso | Script | Qué hace |
|-----:|--------|----------|
| 1 | `preparar_recortes.py` | Recorta la mano de cada imagen de HaGRIDv2 (224 × 224) según las clases de `config/gestos.json` |
| 2 | `entrenar_mobilenet.py` | Ajusta MobileNetV3 por fotograma y lo evalúa en el conjunto de prueba |
| 3 | `entrenar_lstm.py` | Entrena el LSTM con secuencias sintéticas de 16 fotogramas y lo compara con la votación por mayoría |
| 4 | `exportar_tflite.py` | Exporta ambos modelos a TensorFlow Lite y mide la latencia |

Rutas configurables con las variables `GV_HAGRID`, `GV_RECORTES` y
`GV_SALIDA` (ver `entrenamiento/comun.py`). Los resultados se guardan en
`runs/`, que no se sube al repositorio.

Entorno usado: Python 3.10, TensorFlow 2.10.1 con GPU (CUDA 11).

### Primeros resultados (solo HaGRIDv2)

Primera ronda, con 10 de los 13 gestos (Dolor, Beber y Comer no existen en
HaGRID y quedan pendientes de datos propios). La división de HaGRID es por
persona.

| Métrica (por fotograma, 12 000 imágenes de prueba) | Resultado |
|---------|----------:|
| Exactitud | 0.989 |
| Precisión mínima por gesto (Ayuda) | 0.958 |
| Sensibilidad (macro) | 0.992 |
| F1 (macro) | 0.991 |

- El LSTM y la votación por mayoría llegan a 0.999–1.000 sobre secuencias
  sintéticas armadas con imágenes fijas; esto valida el flujo, pero no mide
  el movimiento real. Falta evaluarlo con videos propios.
- Tamaño: 6,0 MB (fotograma) + 107 KB (secuencia).
- Latencia en la CPU del computador: 3,6 ms; falta medirla en el teléfono.
- HaGRID son fotos de adultos bien encuadradas: se espera que el rendimiento
  baje con niños y con la cámara del teléfono.

## Criterios de aceptación (OB2, Tabla 9)

| Métrica | Criterio |
|---------|----------|
| Exactitud global | ≥ 90 % |
| Precisión por gesto | ≥ 90 % |
| Sensibilidad (recall) | ≥ 85 % |
| F1-Score (macro) | ≥ 0.85 |
| Inferencia en el teléfono | ≤ 200 ms |

## Integración con la aplicación

1. Copiar `gestos_fotograma.tflite`, `gestos_secuencia.tflite` y
   `etiquetas.txt` a `assets/models/` de Mobile-App.
2. Los valores de `clase` de `config/gestos.json` deben coincidir con
   `GESTO.clase_reconocimiento` en la aplicación y en el backend.
3. La aplicación acepta un resultado solo si la confianza es de al menos 0,85
   y el gesto está habilitado en el perfil del niño.

## Estado

- [x] Definición del repertorio de gestos (pendiente de validación)
- [ ] Herramienta de grabación en la aplicación
- [ ] Recolección del conjunto de datos
- [x] Scripts de entrenamiento (MobileNetV3 + LSTM) con HaGRIDv2
- [ ] Evaluación contra la Tabla 9 con videos propios
- [x] Exportación a TensorFlow Lite
- [ ] Integración en la aplicación y medición en el teléfono

## Créditos

Este proyecto usa **HaGRIDv2** (A. Nuzhdin, A. Nagaev, A. Sautin,
A. Kapitanov y K. Kvanchiani, 2024,
[arXiv:2412.01508](https://arxiv.org/abs/2412.01508)), versión ampliada de
HaGRID (Kapitanov et al., WACV 2024), publicado bajo la licencia
[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
en <https://github.com/hukenovs/hagrid>. Los modelos de este repositorio se
entrenaron con imágenes de ese conjunto. Las imágenes no se redistribuyen.

---

Proyecto de Taller de Proyecto · Ingeniería de Software · Universidad Peruana
de Ciencias Aplicadas (UPC).
