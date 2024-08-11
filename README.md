[![Github Actions Workflow](https://github.com/DiogoCarapito/hf-mlflow-template/actions/workflows/main.yaml/badge.svg)](https://github.com/DiogoCarapito/hf-mlflow-template/actions/workflows/main.yaml)

# hf-mlflow-template
Template for a Hugging Face fine-tuning classification model, pytorch based with local MLFlow server tracking 

## cheat sheet

### venv
create venv
```bash
python3 -m venv .venv
```

activate venv
```bash
source .venv/bin/activate
```

### Docker
build docker image
```bash
docker build -t main:latest .
```

