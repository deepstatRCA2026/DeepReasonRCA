# DeepReasonRCA: Projection-Based Hybrid Root Cause Analysis for Microservices

This repository implements **DeepReasonRCA**, a lightweight hybrid framework for real-time root cause analysis (RCA) in microservice-based systems.

DeepReasonRCA combines:
- a **lightweight reconstruction-based deep model**, and  
- a **projection-based deviation reasoning module**

to enable accurate, scalable, and efficient fault localization under single- and multi-modality telemetry.

This repository is forked from:
https://github.com/phamquiluan/RCAEval

---

## Overview

DeepReasonRCA is designed for **real-time RCA in production microservices**, where:
- telemetry is high-dimensional,
- failures are sparse and heterogeneous,
- and low-latency inference is required.

Instead of heavy per-variable or per-time-step modeling, DeepReasonRCA:
1. learns compact reconstruction representations of normal behavior
2. performs projection-based residual reasoning for root cause ranking

---

## Evaluation

We evaluate DeepReasonRCA on **7 benchmark datasets** from 3 microservice systems:

- Online Boutique
- Sock Shop
- Train Ticket

### Key results

- Strong improvements over:
  - statistical RCA methods
  - deep learning baselines
  - multimodal fusion methods
- Lightweight models (e.g., DLinear, FITS) gain large improvements when combined with projection-based reasoning

---

## Practical Implications

DeepReasonRCA shows that:
> lightweight reconstruction models + projection-based reasoning  
can outperform heavy deep or purely statistical RCA systems

This makes it suitable for:
- production observability systems
- real-time incident response
- resource-constrained environments

---

## Prerequisites

Experiments were conducted on:

- Intel i9-10900K CPU
- 32GB RAM
- NVIDIA RTX 3070 (8GB)
- Python 3.10+
- PyTorch 2.7.1

---

Install system dependencies:
## Installation

To set up the project environment using Conda, run the following commands in your terminal:

```bash
# 1. Create the environment from the file
conda env create -f environment.yml

# 2. Activate the new environment
conda activate <your-env-name>
```


Then run 
```bash
RQ1.sh
RQ1_multimodal.sh
RQ_Ensemble.sh
```

Then to create the figures, run
```bash
python scripts/RQ_1_Accuracy (single modality).py
python scripts/RQ_1_Accuracy (multi modality).py
python scripts/RQ_2_Ablation (single modality).py
python scripts/RQ_efficiency.py
```