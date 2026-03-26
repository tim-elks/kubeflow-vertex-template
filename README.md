# kubeflow-vertex-template

A production-ready template for building and deploying machine learning pipelines using
**Kubeflow Pipelines SDK** and **Google Cloud Vertex AI Pipelines**.

---

## Overview

This template demonstrates the recommended workflow for ML engineering teams using
the Kubeflow Pipelines v2 SDK to define pipelines and Vertex AI Pipelines as the
managed execution engine.

```
You Write:                         Vertex AI Handles:
  Pipeline Components (Python)  →    Running containers on managed Kubernetes
  Pipeline Definition (KFP SDK) →    Component orchestration
  Docker Images (per component) →    Artifact storage & lineage
                                      Monitoring & logging
```

### Key Distinction

| Concept | Role |
|---|---|
| **Kubeflow Pipelines SDK** | Python library you use to *define* the ML workflow |
| **Vertex AI Pipelines** | Google's managed service that *executes* your pipeline definition |

You don't need a local Kubeflow installation — just the `kfp` Python package.

---

## Repository Structure

```
kubeflow-vertex-template/
├── components/                     # Individual pipeline steps
│   ├── data_preprocessing/
│   │   ├── src/preprocess.py       # Preprocessing logic
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── training/
│   │   ├── src/train.py            # Model training logic
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── evaluation/
│   │   ├── src/evaluate.py         # Model evaluation logic
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── deployment/
│       ├── src/deploy.py           # Vertex AI deployment logic
│       ├── Dockerfile
│       └── requirements.txt
├── pipeline/
│   ├── pipeline.py                 # Main pipeline definition (KFP SDK)
│   └── compile_pipeline.py        # Compiles pipeline to YAML spec
├── scripts/
│   ├── build_components.sh        # Build & push Docker images
│   └── run_pipeline.py            # Submit pipeline to Vertex AI
├── config/
│   └── pipeline_config.yaml       # Project-specific configuration
├── tests/
│   └── test_components.py         # Unit tests for component logic
└── requirements.txt               # Core SDK dependencies
```

---

## Pipeline Steps

The template implements a four-step ML pipeline with a conditional deployment gate:

```
[1] Data Preprocessing
        ↓
[2] Model Training
        ↓
[3] Model Evaluation
        ↓ (only if accuracy ≥ threshold)
[4] Model Deployment → Vertex AI Endpoint
```

| Step | Component | Description |
|---|---|---|
| 1 | `data_preprocessing` | Downloads raw CSV from GCS, drops NaN rows, scales features, splits train/test |
| 2 | `training` | Trains a `RandomForestClassifier`; saves model artifact and training metrics to GCS |
| 3 | `evaluation` | Loads model + test data from GCS; computes accuracy, classification report, confusion matrix |
| 4 | `deployment` | Uploads model to Vertex AI Model Registry and deploys to a Vertex AI Endpoint |

---

## Prerequisites

### Tools
- Python 3.10+
- Docker (for building component images)
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)

### GCP Setup
1. A GCP project with billing enabled
2. Enable the required APIs:
   ```bash
   gcloud services enable \
     aiplatform.googleapis.com \
     storage.googleapis.com \
     containerregistry.googleapis.com
   ```
3. Create a GCS bucket for pipeline artifacts:
   ```bash
   gsutil mb -l us-central1 gs://your-bucket
   ```
4. Authenticate:
   ```bash
   gcloud auth application-default login
   gcloud auth configure-docker
   ```

---

## Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your Project

Copy and edit the configuration file:

```bash
cp config/pipeline_config.yaml config/my_pipeline_config.yaml
# Edit the file — fill in your GCP project, bucket, and dataset paths
```

Key fields to update in `config/pipeline_config.yaml`:

```yaml
gcp:
  project_id: "your-gcp-project-id"
  region: "us-central1"
  pipeline_root: "gs://your-bucket/pipeline-root"

data:
  input_data_uri: "gs://your-bucket/data/raw_data.csv"
  target_column: "label"
```

### 3. Build and Push Component Docker Images

```bash
export GCP_PROJECT=your-gcp-project-id
export IMAGE_TAG=latest
bash scripts/build_components.sh
```

Update `config/pipeline_config.yaml` with the pushed image URIs (the script prints them).

### 4. Compile the Pipeline

```bash
python pipeline/compile_pipeline.py --output pipeline_spec.yaml
```

This generates a `pipeline_spec.yaml` file that Vertex AI understands.

### 5. Submit the Pipeline to Vertex AI

```bash
python scripts/run_pipeline.py --config config/pipeline_config.yaml
```

Monitor progress at:
```
https://console.cloud.google.com/vertex-ai/pipelines
```

---

## Local Development & Testing

### Test Component Logic (no GCP needed)

```bash
# Install test dependencies
pip install pytest pytest-mock scikit-learn pandas google-cloud-storage

# Run all tests
python -m pytest tests/ -v
```

### Test a Single Component Locally

```bash
# Example: run preprocessing locally (needs real GCS URIs or mock data)
python components/data_preprocessing/src/preprocess.py \
  --input-data-uri gs://your-bucket/data/raw_data.csv \
  --train-output-uri gs://your-bucket/data/train.csv \
  --test-output-uri gs://your-bucket/data/test.csv \
  --target-column label
```

### Compile Pipeline Without Submitting

```bash
python pipeline/compile_pipeline.py --output /tmp/pipeline_spec.yaml
```

---

## Customising the Template

### Add a New Pipeline Component

1. Create a new directory under `components/`:
   ```bash
   mkdir -p components/my_step/src
   ```
2. Write your component logic in `components/my_step/src/my_step.py`
3. Add a `Dockerfile` and `requirements.txt`
4. Add a `@dsl.component` wrapper in `pipeline/pipeline.py`
5. Wire it into the `@pipeline` function

### Swap the ML Framework

The training component uses `scikit-learn` by default. To use XGBoost, TensorFlow,
or PyTorch:

1. Update `components/training/requirements.txt`
2. Modify `components/training/src/train.py` with your model code
3. Rebuild and push the Docker image

### Change the Serving Container

Vertex AI provides pre-built serving containers for common frameworks:

| Framework | Container URI |
|---|---|
| scikit-learn 1.4 | `us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-4:latest` |
| XGBoost 1.7 | `us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest` |
| TensorFlow 2.13 | `us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest` |
| PyTorch 1.13 | `us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest` |

Update `serving_container_image_uri` in `config/pipeline_config.yaml`.

---

## Architecture Reference

```
Development Machine
┌──────────────────────────────────────────────┐
│  Python + KFP SDK                            │
│  ┌────────────────┐   compile   ┌──────────┐ │
│  │  pipeline.py   │ ──────────► │ .yaml    │ │
│  └────────────────┘             └────┬─────┘ │
│                                      │ submit │
└──────────────────────────────────────┼───────┘
                                       │
                                       ▼
                          Google Cloud Vertex AI
┌──────────────────────────────────────────────────────┐
│  Vertex AI Pipelines (managed Kubernetes)            │
│                                                      │
│  Step 1: Preprocessing  ──► GCS (train/test CSVs)   │
│  Step 2: Training       ──► GCS (model.pkl)         │
│  Step 3: Evaluation     ──► GCS (metrics.json)      │
│  Step 4: Deployment     ──► Vertex AI Endpoint       │
│                                                      │
│  Each step runs in its own Docker container.         │
└──────────────────────────────────────────────────────┘
```

---

## Learning Path

If you're new to this stack, follow this progression:

**Phase 1 — Local development**
- [ ] Understand Docker basics (each component runs in a container)
- [ ] Install the KFP SDK (`pip install kfp`)
- [ ] Run the component unit tests to understand the data flow
- [ ] Modify `train.py` to use your own model

**Phase 2 — Vertex AI integration**
- [ ] Authenticate with GCP (`gcloud auth application-default login`)
- [ ] Upload a sample dataset to GCS
- [ ] Build and push component images (`scripts/build_components.sh`)
- [ ] Compile and submit the pipeline (`scripts/run_pipeline.py`)
- [ ] Monitor execution in the Vertex AI console

**Optional — Kubernetes internals**
- Explore Minikube only if you want to understand how pipeline steps map to Kubernetes pods.
  This is **not required** for the Kubeflow Pipelines + Vertex AI workflow.

---

## License

[MIT](LICENSE)
