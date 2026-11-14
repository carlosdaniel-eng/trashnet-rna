"""Treinamento de um experimento e registro de histórico, configuração e custo."""
import json
import platform
import time

import keras
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from . import config as C
from .data import class_weights_from_train, load_arrays, load_split
from .models import build_model


def hardware_info() -> dict:
    """Descreve a máquina usada (para a tabela de custo)."""
    gpus = tf.config.list_physical_devices("GPU")
    gpu = tf.config.experimental.get_device_details(gpus[0]).get("device_name") if gpus else "sem GPU"
    return {
        "gpu": gpu,
        "cpu": platform.processor() or platform.machine(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    }


class EpochTimer(keras.callbacks.Callback):
    """Mede o tempo de cada época e guarda o maior uso de RAM observado."""

    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.peak_ram = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self.start)
        self.peak_ram = max(self.peak_ram, psutil.Process().memory_info().rss)


def run_experiment(exp_id: str, force: bool = False) -> dict:
    """Treina um experimento e salva tudo em outputs/<exp_id>/.

    Se o experimento já foi concluído, apenas lê o resumo salvo (útil quando o
    Colab desconecta no meio da sequência). Use force=True para treinar de novo.
    """
    out = C.OUTPUT_DIR / exp_id
    if (out / "summary.json").exists() and not force:
        print(f"{exp_id} já foi treinado. Lendo o resumo salvo.")
        return json.loads((out / "summary.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)

    cfg = C.EXPERIMENTS[exp_id]
    params = C.MLP_PARAMS if cfg["model"] == "mlp" else C.CNN_PARAMS
    tp = C.TRAIN_PARAMS

    # 1) Dados: o carregamento fica fora da medição de tempo.
    split_df = load_split()
    X_train, y_train, _ = load_arrays(split_df, "train", cfg["size"])
    X_val, y_val, _ = load_arrays(split_df, "val", cfg["size"])
    weights = class_weights_from_train(y_train) if cfg["class_weights"] else None

    # 2) Modelo: mesma semente em todos os experimentos.
    keras.backend.clear_session()
    keras.utils.set_random_seed(C.SEED)
    model = build_model(cfg["model"], cfg["size"])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                  loss=tp["loss"], metrics=["accuracy"])
    model.summary()

    # 3) Treino com early stopping pela perda de validação.
    #    O ModelCheckpoint guarda os pesos da melhor época, que são recarregados no final.
    timer = EpochTimer()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=tp["early_stopping_patience"],
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(out / "best.weights.h5", monitor="val_loss",
                                        save_best_only=True, save_weights_only=True),
        timer,
    ]
    if tf.config.list_physical_devices("GPU"):
        tf.config.experimental.reset_memory_stats("GPU:0")
    start = time.perf_counter()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=tp["max_epochs"], batch_size=tp["batch_size"],
                        class_weight=weights, callbacks=callbacks, verbose=2)
    total_time = time.perf_counter() - start
    peak_gpu = None
    if tf.config.list_physical_devices("GPU"):
        peak_gpu = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1024**2
    model.load_weights(out / "best.weights.h5")

    # 4) Salvar histórico e modelo.
    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", range(1, len(hist) + 1))
    hist["epoch_time_s"] = timer.epoch_times
    hist.to_csv(out / "history.csv", index=False)
    best_epoch = int(hist["val_loss"].idxmin()) + 1
    # Salva só a arquitetura e os pesos (sem o estado do otimizador Adam, que
    # triplicaria o tamanho do arquivo). Para isso, copia os pesos para um modelo novo.
    final = build_model(cfg["model"], cfg["size"])
    final.set_weights(model.get_weights())
    final.save(out / "model.keras")

    # 5) Métricas de validação (o teste só é usado no notebook 03).
    y_pred = final.predict(X_val, verbose=0).argmax(axis=1)
    summary = {
        "exp_id": exp_id,
        "rotulo": C.experiment_label(exp_id),
        "modelo": cfg["model"],
        "resolucao_lxa": f"{cfg['size'][0]}x{cfg['size'][1]}",
        "pesos_de_classe": weights,
        "seed": C.SEED,
        "hiperparametros_modelo": params,
        "hiperparametros_treino": tp,
        "parametros": final.count_params(),
        "tamanho_modelo_mb": round((out / "model.keras").stat().st_size / 1024**2, 2),
        "epocas_executadas": len(hist),
        "melhor_epoca": best_epoch,
        "tempo_total_s": round(total_time, 1),
        "tempo_medio_epoca_s": round(float(np.mean(timer.epoch_times)), 2),
        "pico_ram_mb": round(timer.peak_ram / 1024**2),
        "pico_gpu_mb": None if peak_gpu is None else round(peak_gpu),
        "metodo_medicao": "time.perf_counter() em volta do model.fit (sem download e sem leitura "
                          "das imagens); tempo por época inclui a validação; RAM = maior RSS do "
                          "processo ao fim das épocas (psutil); GPU = pico do TensorFlow.",
        "val_accuracy": round(accuracy_score(y_val, y_pred), 4),
        "val_f1_macro": round(f1_score(y_val, y_pred, average="macro", zero_division=0), 4),
        "hardware": hardware_info(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{exp_id}: melhor época {best_epoch} de {len(hist)} | "
          f"F1 macro validação {summary['val_f1_macro']} | {total_time:.0f} s")
    return summary


def summary_table() -> pd.DataFrame:
    """Junta os resumos dos experimentos já treinados em uma tabela."""
    rows = []
    for exp_id in C.EXPERIMENTS:
        f = C.OUTPUT_DIR / exp_id / "summary.json"
        if f.exists():
            rows.append(json.loads(f.read_text(encoding="utf-8")))
    cols = ["exp_id", "rotulo", "parametros", "epocas_executadas", "melhor_epoca", "tempo_total_s",
            "tempo_medio_epoca_s", "pico_ram_mb", "pico_gpu_mb", "val_accuracy", "val_f1_macro"]
    return pd.DataFrame(rows)[cols]
