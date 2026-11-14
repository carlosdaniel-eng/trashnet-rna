"""Métricas, tabelas e figuras exigidas na seção 4 da proposta."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from . import config as C
from .data import load_arrays, load_image

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
                     "axes.titlesize": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
SPLIT_PT = {"train": "treino", "val": "validação", "test": "teste"}


def save_fig(fig, name: str):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Figura salva: {path}")


# ---------------------------------------------------------------------------
# Leitura de artefatos de treino
# ---------------------------------------------------------------------------
def load_summary(exp_id: str) -> dict:
    return json.loads((C.OUTPUT_DIR / exp_id / "summary.json").read_text(encoding="utf-8"))


def load_history(exp_id: str) -> pd.DataFrame:
    return pd.read_csv(C.OUTPUT_DIR / exp_id / "history.csv")


def load_model(exp_id: str):
    import keras
    return keras.models.load_model(C.OUTPUT_DIR / exp_id / "model.keras", compile=False)


# ---------------------------------------------------------------------------
# Previsões e métricas
# ---------------------------------------------------------------------------
def predict_split(exp_id: str, split_df: pd.DataFrame, split: str = "test",
                  batch_size: int = 32) -> pd.DataFrame:
    """Gera (ou lê do disco) as previsões de um modelo salvo, sem novo treinamento."""
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = C.RESULTS_DIR / "previsoes" / f"{split}_{exp_id}.csv"
    if path.exists():
        return pd.read_csv(path)
    size = tuple(C.EXPERIMENTS[exp_id]["size"])
    X, y, part = load_arrays(split_df, split, size)
    probs = load_model(exp_id).predict(X, batch_size=batch_size, verbose=0)
    pred = part[["filepath", "label", "label_idx"]].copy()
    pred["pred_idx"] = probs.argmax(1)
    pred["pred"] = [C.CLASSES[i] for i in pred["pred_idx"]]
    pred["prob_pred"] = probs.max(1)
    for i, c in enumerate(C.CLASSES):
        pred[f"prob_{c}"] = probs[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(path, index=False)
    return pred


def global_metrics(pred: pd.DataFrame) -> dict:
    y, yhat = pred["label_idx"], pred["pred_idx"]
    return {"acuracia": accuracy_score(y, yhat),
            "f1_macro": f1_score(y, yhat, average="macro",
                                 labels=range(C.NUM_CLASSES), zero_division=0)}


def per_class_report(pred: pd.DataFrame) -> pd.DataFrame:
    """Precisão, recall, F1 e suporte por classe (zero_division=0)."""
    p, r, f, s = precision_recall_fscore_support(pred["label_idx"], pred["pred_idx"],
                                                 labels=range(C.NUM_CLASSES), zero_division=0)
    return pd.DataFrame({"classe": C.CLASSES, "precisao": p, "recall": r, "f1": f, "suporte": s})


def confusion(pred: pd.DataFrame) -> np.ndarray:
    return confusion_matrix(pred["label_idx"], pred["pred_idx"], labels=range(C.NUM_CLASSES))


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_class_distribution(split_df: pd.DataFrame):
    t = pd.crosstab(split_df["label"], split_df["split"]).reindex(index=C.CLASSES)[["train", "val", "test"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(C.CLASSES))
    for i, s in enumerate(["train", "val", "test"]):
        bars = ax.bar(x + (i - 1) * 0.27, t[s], width=0.27, label=SPLIT_PT[s])
        ax.bar_label(bars, fontsize=7, padding=1)
    ax.set_xticks(x, [f"{c}\n({C.CLASSES_PT[c]})" for c in C.CLASSES])
    ax.set_ylabel("Imagens")
    ax.set_title("Imagens por classe em cada conjunto")
    ax.legend(frameon=False)
    return fig


def plot_examples(split_df: pd.DataFrame, n_per_class: int = 3, seed: int = C.SEED):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(C.CLASSES), n_per_class, figsize=(2.2 * n_per_class, 1.75 * len(C.CLASSES)))
    for r, c in enumerate(C.CLASSES):
        files = split_df.loc[split_df["label"] == c, "filepath"].to_numpy()
        for k, f in enumerate(rng.choice(files, n_per_class, replace=False)):
            ax = axes[r, k]
            ax.imshow(Image.open(C.DATASET_ROOT / f))
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(c, fontsize=10)
    fig.suptitle("Exemplos de cada classe (512×384)")
    fig.tight_layout()
    return fig


def plot_same_image_resolutions(filepath: str):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, size in zip(axes, C.RESOLUTIONS):
        ax.imshow(load_image(C.DATASET_ROOT / filepath, size))
        ax.set_title(f"{size[0]}×{size[1]}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Mesma imagem nas três resoluções ({filepath})")
    fig.tight_layout()
    return fig


def plot_learning_curves(exp_ids):
    fig, axes = plt.subplots(len(exp_ids), 2, figsize=(9, 2.1 * len(exp_ids)), squeeze=False)
    for r, e in enumerate(exp_ids):
        h = load_history(e)
        best = int(h["val_loss"].idxmin()) + 1
        for ax, (m, nome) in zip(axes[r], [("loss", "Perda"), ("accuracy", "Acurácia")]):
            ax.plot(h["epoch"], h[m], label="treino")
            ax.plot(h["epoch"], h[f"val_{m}"], label="validação")
            ax.axvline(best, color="gray", ls="--", lw=0.8, label=f"melhor época ({best})")
            ax.set_title(f"{C.experiment_label(e)} — {nome}")
            ax.set_xlabel("Época")
        axes[r, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def plot_confusion(ax, cm: np.ndarray, title: str):
    """Contagem e porcentagem por linha na mesma célula; reais nas linhas."""
    row = cm.sum(1, keepdims=True)
    pct = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0) * 100
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if pct[i, j] > 55 else "black")
    ax.set_xticks(range(len(C.CLASSES)), C.CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(C.CLASSES)), C.CLASSES)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(title)


def plot_confusions(preds: dict):
    n = len(preds)
    cols = 3 if n > 2 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    for ax, (e, pred) in zip(axes.flat, preds.items()):
        plot_confusion(ax, confusion(pred), C.experiment_label(e) if e in C.EXPERIMENTS else e)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_resolution_effect(res: pd.DataFrame):
    """res: colunas resolucao, acuracia, f1_macro, tempo_medio_epoca_s, tempo_total_s."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(res))
    b = a1.bar(x - 0.2, res["acuracia"], 0.4, label="acurácia")
    a1.bar_label(b, fmt="%.3f", fontsize=7)
    b = a1.bar(x + 0.2, res["f1_macro"], 0.4, label="F1 macro")
    a1.bar_label(b, fmt="%.3f", fontsize=7)
    a1.set_xticks(x, res["resolucao"]); a1.set_ylim(0, 1)
    a1.set_title("CNN sem pesos — desempenho no teste"); a1.legend(frameon=False)
    b = a2.bar(x - 0.2, res["tempo_medio_epoca_s"], 0.4, label="tempo médio por época (s)")
    a2.bar_label(b, fmt="%.1f", fontsize=7)
    a2.set_ylabel("Segundos por época")
    a2b = a2.twinx()
    a2b.plot(x, res["tempo_total_s"] / 60, "o-", color="C3", label="tempo total (min)")
    a2b.set_ylabel("Tempo total (min)")
    a2b.spines["right"].set_visible(True)
    a2.set_xticks(x, res["resolucao"])
    a2.set_title("CNN sem pesos — custo de treinamento")
    h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a2b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def plot_errors(pred: pd.DataFrame, title: str, n: int = 6):
    """Até n imagens de teste classificadas incorretamente, com real, prevista e probabilidade."""
    wrong = pred[pred["label_idx"] != pred["pred_idx"]]
    if wrong.empty:
        print("O modelo não errou nenhuma imagem do teste.")
        return None, wrong
    # mostra os erros com maior confiança (mais informativos para a discussão)
    sel = wrong.sort_values("prob_pred", ascending=False).head(n)
    cols = min(3, len(sel))
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, sel.iterrows()):
        ax.imshow(Image.open(C.DATASET_ROOT / r["filepath"]))
        # nome do arquivo no título (em vez de abaixo da imagem) para não sobrepor a linha seguinte
        ax.set_title(f"{r['filepath'].split('/')[-1]}\nreal: {r['label']} | prevista: {r['pred']} "
                     f"({r['prob_pred']:.1%})", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, wrong
