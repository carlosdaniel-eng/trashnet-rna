"""Download, inventário, divisão dos dados e leitura das imagens."""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config as C


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_dataset() -> Path:
    """Baixa e extrai o dataset-resized.zip, se ainda não estiver na pasta de dados."""
    if C.DATASET_ROOT.exists():
        print("Dataset já disponível em", C.short_path(C.DATASET_ROOT))
        return C.DATASET_ROOT
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not C.DATASET_ZIP.exists():
        print("Baixando", C.DATASET_URL)
        urllib.request.urlretrieve(C.DATASET_URL, C.DATASET_ZIP)
    with zipfile.ZipFile(C.DATASET_ZIP) as z:
        z.extractall(C.DATA_DIR)
    print("Dataset extraído em", C.short_path(C.DATASET_ROOT))
    return C.DATASET_ROOT


# ---------------------------------------------------------------------------
# Inventário: uma linha por imagem
# ---------------------------------------------------------------------------
def build_inventory() -> pd.DataFrame:
    """Lista as imagens de cada classe com dimensões, modo de cor, integridade e MD5."""
    rows = []
    for label in C.CLASSES:
        for path in sorted((C.DATASET_ROOT / label).glob("*.jpg")):
            row = {"filepath": f"{label}/{path.name}", "label": label, "ok": True,
                   "width": None, "height": None, "mode": None}
            try:
                with Image.open(path) as im:
                    im.load()  # força a leitura completa: falha se o arquivo estiver corrompido
                    row["width"], row["height"], row["mode"] = im.width, im.height, im.mode
            except Exception:
                row["ok"] = False
            # MD5 do arquivo: dois arquivos com o mesmo MD5 são cópias idênticas
            row["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
            rows.append(row)
    inv = pd.DataFrame(rows)
    inv["label_idx"] = inv["label"].map(C.CLASS_TO_IDX)
    return inv


def find_duplicates(inv: pd.DataFrame) -> pd.DataFrame:
    """Arquivos que aparecem mais de uma vez (mesmo MD5)."""
    dup = inv[inv.duplicated("md5", keep=False)]
    return dup.sort_values("md5")[["md5", "filepath", "label"]]


# ---------------------------------------------------------------------------
# Divisão 70 / 15 / 15 estratificada
# ---------------------------------------------------------------------------
def make_split(inv: pd.DataFrame) -> pd.DataFrame:
    """Divide as imagens em treino, validação e teste mantendo a proporção das classes."""
    df = inv[inv["ok"] & ~inv["filepath"].isin(C.ARQUIVOS_EXCLUIDOS)].copy()
    print("Imagens removidas antes da divisão:", len(inv) - len(df))

    # 1º corte: 70% treino e 30% restante; 2º corte: o restante vira 15% validação e 15% teste.
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=C.SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=C.SEED)
    train, val, test = train.assign(split="train"), val.assign(split="val"), test.assign(split="test")
    split_df = pd.concat([train, val, test]).sort_values(["split", "label_idx", "filepath"])

    # Verificação de vazamento: nenhuma imagem idêntica pode estar em dois conjuntos.
    assert split_df.groupby("md5")["split"].nunique().max() == 1, \
        "Há cópias idênticas em conjuntos diferentes; inclua uma delas em ARQUIVOS_EXCLUIDOS."
    return split_df[["filepath", "label", "label_idx", "split", "md5"]].reset_index(drop=True)


def save_split(split_df: pd.DataFrame) -> None:
    C.SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(C.SPLIT_CSV, index=False)
    print("Divisão salva em", C.short_path(C.SPLIT_CSV))


def load_split() -> pd.DataFrame:
    if not C.SPLIT_CSV.exists():
        raise FileNotFoundError("splits/split.csv não encontrado. Rode o notebook 01 primeiro.")
    return pd.read_csv(C.SPLIT_CSV)


def split_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    """Tabela com a quantidade de imagens de cada classe em cada conjunto."""
    t = pd.crosstab(split_df["label"], split_df["split"]).reindex(index=C.CLASSES)
    t = t[["train", "val", "test"]]
    t["total"] = t.sum(axis=1)
    t.loc["total"] = t.sum()
    return t


# ---------------------------------------------------------------------------
# Leitura das imagens
# ---------------------------------------------------------------------------
def load_image(path, size) -> np.ndarray:
    """Abre uma imagem em RGB e redimensiona para size = (largura, altura).

    Retorna um array uint8 (altura, largura, 3) com valores de 0 a 255.
    A divisão por 255 é feita dentro do modelo (camada Rescaling).
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != tuple(size):
            im = im.resize(tuple(size), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def load_arrays(split_df: pd.DataFrame, split: str, size):
    """Carrega todas as imagens de um conjunto ('train', 'val' ou 'test') em memória."""
    part = split_df[split_df["split"] == split].reset_index(drop=True)
    X = np.stack([load_image(C.DATASET_ROOT / f, size) for f in part["filepath"]])
    y = part["label_idx"].to_numpy()
    return X, y, part


def class_weights_from_train(y_train: np.ndarray) -> dict:
    """Pesos inversamente proporcionais à frequência de cada classe no treino.

    peso da classe c = total de imagens / (número de classes * imagens da classe c)
    """
    counts = np.bincount(y_train, minlength=C.NUM_CLASSES)
    total = len(y_train)
    return {c: total / (C.NUM_CLASSES * counts[c]) for c in range(C.NUM_CLASSES)}
