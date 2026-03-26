# Kubeflow Pipelines + Vertex AI Template

Template repository for creating and deploying ML applications using **Kubeflow Pipelines SDK** on **Vertex AI Pipelines**.

---

## What This Means

- You **write** Kubeflow Pipeline code (Python SDK)
- But you **run** it on Vertex AI Pipelines (managed service)
- **Best of both worlds**: familiar framework, managed infrastructure

### Key Distinction

| Layer | Tool | Responsibility |
|---|---|---|
| Pipeline Definition | Kubeflow Pipelines SDK | Your code (Python) |
| Execution Engine | Vertex AI Pipelines | Google's managed infrastructure |
| Component Packaging | Docker | Containerize each step |

> You don't need full Kubeflow installed — just the **Kubeflow Pipelines SDK** (`pip install kfp`).

---

## Architecture

```
You Write:
  Pipeline Component (Python) → Dockerized
  Pipeline Definition (Kubeflow SDK)

Vertex AI Handles:
  - Running containers on Kubernetes (managed)
  - Component orchestration
  - Artifact storage
  - Monitoring/logging
```

---

## Project Structure

```
kubeflow-vertex-template/
├── components/                   # Individual pipeline components
│   ├── data_ingestion/
│   │   ├── component.py          # Component logic
│   │   ├── Dockerfile            # Container definition
│   │   └── requirements.txt      # Component dependencies
│   ├── preprocessing/
│   │   ├── component.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── training/
│       ├── component.py
│       ├── Dockerfile
│       └── requirements.txt
├── pipeline/
│   └── pipeline.py               # KFP pipeline definition (assembles components)
├── vertex_ai/
│   ├── submit_pipeline.py        # Compile and submit pipeline to Vertex AI
│   └── config.yaml               # GCP project configuration
├── local/
│   └── run_local.py              # Test pipeline logic locally (no Kubernetes needed)
├── requirements.txt              # SDK and dev dependencies
└── .gitignore
```

---

## Recommended Learning Path

### Phase 1 — Local Development

1. ✅ **Docker basics** — containerize model code
2. ✅ **Install Kubeflow Pipelines SDK** — `pip install kfp`
3. ✅ **Build simple pipeline components** — see `components/`
4. ✅ **Test pipeline logic locally** — see `local/run_local.py` (no Kubernetes needed)

### Phase 2 — Vertex AI Integration

1. ✅ **Authenticate with GCP** — `gcloud auth application-default login`
2. ✅ **Compile pipeline to Vertex AI format** — `python pipeline/pipeline.py`
3. ✅ **Submit pipeline runs to Vertex AI** — `python vertex_ai/submit_pipeline.py`
4. ✅ **Monitor execution** in Vertex AI console

### Optional — Minikube + Kubeflow

> Only needed if you want to understand pod/container orchestration internals.
> **Not required** for Kubeflow Pipelines + Vertex AI workflow.

---

## Quick Start

### Prerequisites

- Python 3.9+
- Docker
- GCP project with Vertex AI API enabled
- `gcloud` CLI authenticated

### Setup

```bash
# Clone this template
git clone https://github.com/tim-elks/kubeflow-vertex-template.git
cd kubeflow-vertex-template

# Install dependencies
pip install -r requirements.txt
```

### Local Testing (No GCP Required)

```bash
python local/run_local.py
```

### Build and Push Component Images

```bash
# Set your GCP project and Artifact Registry
export PROJECT_ID=your-project-id
export REGION=us-central1
export REPO=your-artifact-registry-repo

# Build and push each component
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/data-ingestion:latest \
  components/data_ingestion/
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/data-ingestion:latest

docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/preprocessing:latest \
  components/preprocessing/
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/preprocessing:latest

docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/training:latest \
  components/training/
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/training:latest
```

### Configure and Submit to Vertex AI

Edit `vertex_ai/config.yaml` with your project details, then:

```bash
# Compile the pipeline spec
uv run python pipeline/pipeline.py

# Submit to Vertex AI
uv run python vertex_ai/submit_pipeline.py
```

---

## Tool Relationships

### Docker
- Packages each pipeline component
- Each pipeline step runs in its own container

### Kubeflow Pipelines SDK
- Python library to define ML workflows
- Creates pipeline YAML/JSON specs
- Same code works locally or on Vertex AI

### Vertex AI Pipelines
- Executes your Kubeflow pipeline definitions
- Managed Kubernetes backend (you don't see it)
- Integrated with other Vertex AI services (training, serving)

### Minikube (Optional)
- Useful for understanding how pipeline steps run as pods
- **Not required** if you go straight to Vertex AI

---

## License

MIT
