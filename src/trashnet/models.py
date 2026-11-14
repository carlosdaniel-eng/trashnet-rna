"""Arquiteturas MLP e CNN (Keras)."""
import keras
from keras import layers

from . import config as C


def build_mlp(size, hidden_units, dropout):
    """MLP: os pixels RGB viram um vetor e passam por camadas densas com ReLU."""
    w, h = size
    inputs = keras.Input(shape=(h, w, 3))           # no Keras: altura x largura x canais
    x = layers.Rescaling(1.0 / 255)(inputs)         # normalização: pixels de 0-255 para 0-1
    x = layers.Flatten()(x)                         # imagem -> vetor
    for units in hidden_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(C.NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mlp")


def build_cnn(size, filters, kernel_size, dense_units, dropout):
    """CNN: blocos de convolução + max pooling, global average pooling e camada densa.

    O global average pooling tira a média de cada mapa de características, então a
    saída dele tem sempre o mesmo tamanho (= número de filtros do último bloco),
    seja qual for a resolução. Assim a mesma rede, com o mesmo número de
    parâmetros, é usada em 128x96, 256x192 e 512x384.
    """
    w, h = size
    inputs = keras.Input(shape=(h, w, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    for f in filters:
        x = layers.Conv2D(f, kernel_size, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)               # reduz altura e largura pela metade
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(C.NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn")


def build_model(model_type, size):
    if model_type == "mlp":
        p = C.MLP_PARAMS
        return build_mlp(size, p["hidden_units"], p["dropout"])
    p = C.CNN_PARAMS
    return build_cnn(size, p["filters"], p["kernel_size"], p["dense_units"], p["dropout"])
