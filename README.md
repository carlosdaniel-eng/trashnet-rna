# Classificação de resíduos por imagens (TrashNet) — MLP × CNN

Atividade final de **Redes Neurais Artificiais** — EAD Unifor.

> **Autor:** Carlos Daniel Albino Alves (matrícula 2523582)
> **Artigo:** [`artigo/artigo_trashnet.pdf`](artigo/artigo_trashnet.pdf)

Estudo de classificação de imagens de resíduos em seis classes (glass, paper, cardboard, plastic, metal, trash) com o dataset
TrashNet. Ele compara MLP e CNN, investiga o efeito de **pesos de classe** sobre o desequilíbrio e avalia se **resoluções
maiores** compensam o custo de treinamento da CNN.

| Exp. | Modelo | Resolução (L × A) | Pesos de classe |
|---|---|---|---|
| E1_mlp_128x96 | MLP | 128 × 96 | Não |
| E2_mlp_128x96_pesos | MLP | 128 × 96 | Sim |
| E3_cnn_128x96 | CNN | 128 × 96 | Não |
| E4_cnn_128x96_pesos | CNN | 128 × 96 | Sim |
| E5_cnn_256x192 | CNN | 256 × 192 | Não |
| E6_cnn_512x384 | CNN | 512 × 384 | Não |
| Extra (opcional) | YOLO11n-cls pré-treinado | 224 × 224 | Não |

## Estrutura do repositório

```
trashnet-rna/
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb   # inspeção, integridade, duplicatas, exemplos e divisão 70/15/15
│   ├── 02_treinamento.ipynb            # os seis treinos (GPU)
│   ├── 03_resultados.ipynb             # métricas de teste, tabelas, figuras e análise de erros (sem treinar)
│   └── 04_extra_yolo.ipynb             # objetivo extra com YOLO (opcional)
├── src/trashnet/
│   ├── config.py      # classes, semente, caminhos, experimentos e hiperparâmetros
│   ├── data.py        # download, inventário, divisão estratificada, pré-processamento, pesos de classe
│   ├── models.py      # arquiteturas MLP e CNN (Keras)
│   ├── train.py       # treinamento, early stopping, registro de histórico, configuração e custo
│   └── evaluate.py    # métricas, matrizes de confusão, curvas e gráficos
├── splits/
│   ├── split.csv        # divisão usada em TODOS os experimentos (arquivo, classe, conjunto, md5)
│   └── inventario.csv   # inventário completo (dimensões, modo de cor, integridade, md5)
├── outputs/<exp_id>/  # model.keras, history.csv, summary.json
├── outputs/yolo/      # treino do YOLO (best.pt, results.csv, custo.json, curvas)
├── results/           # tabelas (.csv) e previsões de validação/teste
├── figures/           # figuras usadas no artigo
└── requirements.txt
```

## Como executar no Google Colab (recomendado)

1. Envie a pasta `trashnet-rna` inteira para o Google Drive (em qualquer pasta).
   Na primeira célula de cada notebook, coloque em `PROJECT_DIR` o caminho dessa pasta no Drive
   (painel *Arquivos* do Colab → botão direito na pasta → *Copiar caminho*).
2. Abra os notebooks pelo Drive (botão direito → *Abrir com* → *Google Colaboratory*) e rode **na ordem**:
   - `01_analise_exploratoria` — CPU basta. Baixa o dataset para `/content/data` e confere/gera `splits/split.csv`.
   - `02_treinamento` — **ative a GPU** (*Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU*).
     Cada experimento fica em uma célula própria. Se a sessão cair, rode de novo: experimentos já concluídos são pulados.
   - `03_resultados` — gera todas as tabelas e figuras da seção 4 da proposta.
   - `04_extra_yolo` — opcional (bônus), também com GPU.
3. Tudo o que é gerado (modelos, históricos, tabelas e figuras) é salvo no Drive, dentro da pasta do projeto.

O Colab já possui TensorFlow/Keras, scikit-learn, pandas, matplotlib, Pillow e psutil; o notebook 04 instala o `ultralytics`.

### Avaliação sem novo treinamento

Com os modelos em `outputs/<exp_id>/model.keras` (treinados por vocês ou baixados do link abaixo), basta rodar
`03_resultados.ipynb`: ele apenas carrega os modelos salvos e calcula as previsões, métricas e figuras.

**Modelos treinados:** já estão neste repositório: `outputs/E*_.../model.keras` (os seis experimentos) e
`outputs/yolo/yolo11n_cls_224/weights/best.pt` (YOLO). Todos têm menos de 100 MB, então não é preciso baixar nada à parte.

### Execução local (opcional)

```bash
pip install -r requirements.txt
```

Abra os notebooks a partir da pasta `notebooks/` (Jupyter ou VS Code); os dados são baixados para `data/`.
Versões testadas: Python 3.10+, TensorFlow ≥ 2.16 (Keras 3), scikit-learn ≥ 1.3.

## Decisões metodológicas (resumo; detalhar e justificar no artigo)

**Dados**
- Fonte: `dataset-resized.zip` do repositório TrashNet (Hugging Face), 2.527 imagens JPEG RGB de 512 × 384, sem arquivos corrompidos
  e sem rotação EXIF.
- **Duplicatas:** a verificação por MD5 encontrou **3 pares de arquivos idênticos com rótulos diferentes**
  (`glass115 = metal91`, `glass176 = plastic152`, `glass389 = plastic332`). Como não é possível saber o rótulo correto, os
  6 arquivos foram removidos (lista `ARQUIVOS_EXCLUIDOS` em `config.py`), restando **2.521 imagens**. O notebook 01 confere que
  nenhuma cópia idêntica ficou em dois conjuntos diferentes.
- **Divisão:** estratificada por classe, 70/15/15, semente 42 → 1.764 treino, 378 validação, 379 teste (`splits/split.csv`).
- **Pré-processamento:** conversão para RGB e redimensionamento a partir da versão 512 × 384 (Pillow, interpolação bilinear).
  A normalização (**pixels ÷ 255**) é uma camada `Rescaling` dentro do modelo, de modo que treino, validação, teste e os modelos
  salvos usam exatamente o mesmo pré-processamento. Não foi usado aumento de dados nos seis experimentos obrigatórios.
- **Pesos de classe (E2, E4):** inversamente proporcionais à frequência, calculados só no treino: `w_c = N / (K · n_c)`
  (trash ≈ 3,06; paper ≈ 0,71).

**Modelos** (`src/trashnet/models.py`; entrada Keras = altura × largura × canais)
- **MLP:** Rescaling → Flatten (128·96·3 = 36.864 entradas) → Dense 256 ReLU → Dropout 0,3 → Dense 128 ReLU → Dropout 0,3 → Dense 6 softmax.
- **CNN:** Rescaling → 4 blocos [Conv2D 3×3 ReLU (32, 64, 128, 256 filtros) + MaxPooling 2×2] → **Global Average Pooling** →
  Dropout 0,3 → Dense 64 ReLU → Dropout 0,3 → Dense 6 softmax. Com o GAP, o número de parâmetros é o mesmo nas três resoluções.

**Treinamento** (`src/trashnet/config.py`)
- Perda: entropia cruzada categórica esparsa; otimizador: Adam; taxa de aprendizagem 1e-4 (MLP) e 1e-3 (CNN); lote 32;
  até 100 épocas; **early stopping** pela perda de validação (paciência 10) com recuperação dos melhores pesos; semente 42.
- Entre E1/E2 e entre E3/E4 muda **apenas** o uso de pesos de classe; entre E3/E5/E6 muda **apenas** a resolução (mesma
  CNN, hardware, semente, otimizador, taxa, lote e política de parada; cada treino começa com pesos novos).
- **Seleção:** F1 macro na validação. O teste só é usado no notebook 03, depois de todas as escolhas.

**Custo:** parâmetros, tamanho do `.keras` salvo (sem estado do otimizador), épocas, tempo total de `model.fit` e tempo médio por
época (`time.perf_counter`, excluindo download e leitura das imagens), maior uso de RAM do processo (psutil) e pico de memória da GPU
(`tf.config.experimental.get_memory_info`). Tudo fica registrado em `outputs/<exp_id>/summary.json`, junto com o hardware.

## Referências

- YANG, M.; THUNG, G. **Classification of trash for recyclability status**. Stanford: Stanford University, 2016. Relatório de projeto (CS229).
  Disponível em: https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf.
- THUNG, G. **TrashNet**. GitHub, 2017. Disponível em: https://github.com/garythung/trashnet.
- ABADI, M. et al. **TensorFlow**: large-scale machine learning on heterogeneous systems, 2015. Disponível em: https://www.tensorflow.org.
- CHOLLET, F. et al. **Keras**, 2015. Disponível em: https://keras.io.
- PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825–2830, 2011.
- JOCHER, G.; QIU, J. **Ultralytics YOLO11**, 2024. Disponível em: https://github.com/ultralytics/ultralytics.

## Uso de IA

O código deste repositório foi desenvolvido com apoio de IA generativa (permitido pela proposta). As decisões metodológicas,
a análise e a interpretação dos resultados são do autor.
