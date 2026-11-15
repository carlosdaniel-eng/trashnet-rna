"""Configurações centrais do estudo (classes, caminhos, experimentos e hiperparâmetros).

Todas as dimensões são escritas como (largura, altura), como na proposta.
Ao montar tensores para o Keras, a ordem passa a ser (altura, largura, canais).
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Classes (ordem exigida nas matrizes de confusão e usada como índice dos rótulos)
# ---------------------------------------------------------------------------
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
CLASSES_PT = {
    "glass": "vidro",
    "paper": "papel",
    "cardboard": "papelão",
    "plastic": "plástico",
    "metal": "metal",
    "trash": "outros",
}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------------------------
# Reprodutibilidade e divisão
# ---------------------------------------------------------------------------
SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# DECISÃO DA EQUIPE: arquivos removidos antes da divisão.
# A análise exploratória encontrou 3 pares de arquivos idênticos (mesmo MD5)
# com rótulos diferentes. Como não dá para saber qual rótulo está certo, a
# escolha inicial foi remover as duas cópias de cada par. Se a equipe decidir
# manter uma das cópias, basta apagar o nome dela desta lista.
ARQUIVOS_EXCLUIDOS = [
    "glass/glass115.jpg", "metal/metal91.jpg",
    "glass/glass176.jpg", "plastic/plastic152.jpg",
    "glass/glass389.jpg", "plastic/plastic332.jpg",
]

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
# No Colab, os dados ficam no disco local da máquina (/content/data), que é bem
# mais rápido que o Google Drive. Pode ser alterado pela variável de ambiente.
DATA_DIR = Path(os.environ.get("TRASHNET_DATA_DIR", PROJECT_DIR / "data"))
DATASET_URL = (
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/"
    "dataset-resized.zip?download=true"
)
DATASET_ZIP = DATA_DIR / "dataset-resized.zip"
DATASET_ROOT = DATA_DIR / "dataset-resized"

SPLIT_CSV = PROJECT_DIR / "splits" / "split.csv"
INVENTORY_CSV = PROJECT_DIR / "splits" / "inventario.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"      # modelos, históricos, configurações
RESULTS_DIR = PROJECT_DIR / "results"     # tabelas e previsões
FIGURES_DIR = PROJECT_DIR / "figures"     # figuras do artigo

# ---------------------------------------------------------------------------
# Resoluções (largura, altura). A versão de referência é 512 x 384.
# ---------------------------------------------------------------------------
REFERENCE_SIZE = (512, 384)
RESOLUTIONS = [(128, 96), (256, 192), (512, 384)]

# ---------------------------------------------------------------------------
# DECISÃO DA EQUIPE: hiperparâmetros de cada arquitetura.
# São os mesmos com e sem pesos de classe e nas três resoluções da CNN, para
# que cada comparação mude uma coisa só.
# ---------------------------------------------------------------------------
MLP_PARAMS = {
    "hidden_units": [256, 128],
    "dropout": 0.3,
    "learning_rate": 1e-4,
}
CNN_PARAMS = {
    "filters": [32, 64, 128, 256],  # um bloco conv + max pooling por item
    "kernel_size": 3,
    "dense_units": 64,
    "dropout": 0.3,
    "learning_rate": 1e-3,
}
TRAIN_PARAMS = {
    "optimizer": "adam",
    "loss": "sparse_categorical_crossentropy",
    "batch_size": 32,
    "max_epochs": 100,
    "early_stopping_monitor": "val_loss",
    "early_stopping_patience": 10,
}

# ---------------------------------------------------------------------------
# Os seis experimentos obrigatórios
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "E1_mlp_128x96": {"model": "mlp", "size": (128, 96), "class_weights": False},
    "E2_mlp_128x96_pesos": {"model": "mlp", "size": (128, 96), "class_weights": True},
    "E3_cnn_128x96": {"model": "cnn", "size": (128, 96), "class_weights": False},
    "E4_cnn_128x96_pesos": {"model": "cnn", "size": (128, 96), "class_weights": True},
    "E5_cnn_256x192": {"model": "cnn", "size": (256, 192), "class_weights": False},
    "E6_cnn_512x384": {"model": "cnn", "size": (512, 384), "class_weights": False},
}


def short_path(path) -> str:
    """Caminho relativo à pasta do projeto, quando possível (mensagens independentes da máquina)."""
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def experiment_label(exp_id: str) -> str:
    """Rótulo curto e legível para tabelas e gráficos."""
    cfg = EXPERIMENTS[exp_id]
    w, h = cfg["size"]
    pesos = "com pesos" if cfg["class_weights"] else "sem pesos"
    return f"{cfg['model'].upper()} {w}×{h} {pesos}"
