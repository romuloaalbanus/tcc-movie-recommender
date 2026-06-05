# TCC — Sistema de Recomendação de Filmes

## Requisitos

- Python 3.11

## Instalação

```bash
cd TCC
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Parte 1 — Script de avaliação

```bash
python evaluate.py
```

Gera os arquivos em `results/`:
- `metricas_rmse_mae.csv`
- `metricas_precision_recall.csv`
- `grafico_rmse_mae.png`
- `grafico_precision_k.png`
- `grafico_recall_k.png`

## Parte 2 — Aplicação web

```bash
python app.py
```

Acesse `http://localhost:5000` no navegador.
