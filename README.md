# 💳 Customer Segmentation Platform – End-to-End ML System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Clustering](https://img.shields.io/badge/ML-Unsupervised-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-black)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-success)

> Production-ready customer segmentation system using unsupervised machine learning to identify behavioral clusters in credit card data.

# 📌 Business Context

Banks and fintech companies need to:

* Identify high-value customers
* Detect risky spending behavior
* Personalize credit offers
* Optimize marketing spend
* Improve customer retention

This project builds a **scalable ML segmentation pipeline** to solve that.

# 🧠 System Architecture
<img width="3615" height="164" alt="mermaid-diagram (1)" src="https://github.com/user-attachments/assets/f3e239cf-e4af-4367-bbbb-dae1e595a637" />

# 🔎 1️⃣ Data Engineering

### Preprocessing

* Missing value imputation
* Outlier detection
* Log transformation for skewed features
* Standard scaling

### Feature Engineering

* Credit utilization ratio
* Purchase frequency metrics
* Cash advance ratio
* Payment consistency index

# 🤖 2️⃣ Model Development

### Algorithms Evaluated

| Model        | Silhouette Score | Stability |
| ------------ | ---------------- | --------- |
| K-Means      | **0.42**         | High      |
| Hierarchical | 0.39             | Medium    |
| DBSCAN       | 0.21             | Low       |

Final model: **K-Means (k=4)**

### Cluster Selection Strategy

* Elbow Method
* Silhouette Analysis
* Business interpretability validation

# 🏷 3️⃣ Identified Customer Segments

### 🥇 Premium High Spenders

High balance, high purchases, strong credit limits.

### 🥈 Stable Customers

Moderate spending, consistent payments.

### 🥉 Cash Advance Heavy Users

High withdrawals, potential financial risk.

### 💤 Low Engagement Users

Low activity, low revenue contribution.

# 📊 4️⃣ Evaluation Strategy

* Silhouette Score
* Inertia
* PCA visualization
* Cluster stability tests
* Feature importance per cluster

# 🏗 5️⃣ End-to-End ML Pipeline

Implemented using sklearn Pipeline:

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.90)),
    ('cluster', KMeans(n_clusters=4))
])
```

Pipeline supports:

* Re-training
* Batch scoring
* Real-time inference
* Reproducibility

# 🌐 6️⃣ Deployment Architecture

## 🔹 REST API (FastAPI)

Endpoints:

* `/predict` → Assign cluster
* `/health` → Check service
* `/retrain` → Trigger retraining

## 🔹 Dockerized Service

```bash
docker build -t segmentation-api .
docker run -p 8000:8000 segmentation-api
```

# 📊 7️⃣ Monitoring & MLOps

* Cluster drift detection
* Distribution monitoring
* Re-clustering trigger logic
* Versioned models
* Experiment tracking (optional MLflow)
