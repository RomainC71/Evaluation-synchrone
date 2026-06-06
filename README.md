# Churn prediction MLOps

Projet de prédiction de churn client.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sous Windows, l'environnement virtuel peut être activé avec la commande adaptée à PowerShell.

## Exécution du pipeline

```bash
python main.py
```

Le pipeline entraîne un modèle et écrit les artefacts dans `artifacts/`.

## API

```bash
uvicorn app:app --reload --port 8000
```

Routes disponibles :

- `/health`
- `/predict`
- `/metrics`

## Tests

```bash
pytest
```
