# Avaliação Semântica de Super-Resolução Temporal em Imagens Sentinel-2

Trabalho final da disciplina **CMP617 – Processamento de Imagens** (PPGC/UFRGS).

## Resumo

Pipeline de avaliação semântica que usa métricas de PLN e modelos visão-linguagem (BLIP, GeoChat, CLIP, RemoteCLIP) para medir ganhos de qualidade em imagens Sentinel-2 super-resolvidas por fusão temporal, sem depender de métricas por pixel. A contribuição central cruza estimativas VQA com um ground truth vetorial (GeoPackage), produzindo erro em pontos percentuais interpretável por classe de cobertura do solo.

## Estrutura

```
artigo/          # Fonte LaTeX e PDF do artigo
apresentacao.pdf # Slides da apresentação
codigo/          # Scripts Python do pipeline
dados_vetoriais/ # GeoJSONs de cobertura do solo (7 classes)
resultados/      # CSVs e figuras por região (r1, r2) e consolidado
```

## Código

### Requisitos

```bash
pip install -r codigo/requirements.txt
```

### Execução do pipeline

```bash
bash codigo/run_pipeline.sh
```

Os scripts individuais seguem a ordem:

| Script | Descrição |
|--------|-----------|
| `prepare_data.py` | Reprojeção, tiling e atribuição de classe |
| `blip_nlp_metrics.py` | Legendagem BLIP + métricas NLP |
| `blip_clipscore_metrics.py` | CLIPScore com BLIP |
| `geochat_nlp_metrics.py` | Legendagem GeoChat + métricas NLP |
| `geochat_vqa_metrics.py` | VQA estratificada (7 perguntas, 3 níveis) |
| `geochat_vqa_vetorial.py` | Avaliação cruzada VQA × GeoPackage |
| `geochat_clipscore_metrics.py` | CLIPScore imagem-imagem (CLIP e RemoteCLIP) |
| `consolidar_regioes.py` | Combina resultados das duas regiões |
| `gerar_figuras_consolidadas.py` | Gera figuras dos resultados consolidados |

## Resultados principais

| Métrica | LR | SR | Δ | SR > LR |
|---------|----|----|---|---------|
| RemoteCLIP img-img | 1,532 | 2,040 | **+0,508** | 952/953 tiles |
| MAE global (VQA × GT vetorial) | 22,7 pp | 19,1 pp | **−3,6 pp** | — |
| MAE bare soil | 27,4 pp | 16,4 pp | **+11,1 pp** | — |
| MAE agriculture | 38,3 pp | 31,9 pp | **+6,3 pp** | — |

Validado em **953 tiles** de duas regiões geograficamente distintas do Sul do Brasil.
