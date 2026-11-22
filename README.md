<div align="center">

# 🚘 SRMARL

### Safety-Regulated Curriculum Multi-Agent Reinforcement Learning for Cooperative On-Ramp Merging in Heterogeneous Traffic

**Learning to merge cooperatively under incomplete observations and tail risks.**

[![IEEE TITS](https://img.shields.io/badge/IEEE-TITS%202026-00629B)](https://doi.org/10.1109/TITS.2026.3729124)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1109%2FTITS.2026.3729124-blue)](https://doi.org/10.1109/TITS.2026.3729124)

**Jingjia Teng · Yang Li · Wenjie Huang · Jiawei Wang**  
**Hongmao Qin · Manjiang Hu · Helai Huang · Bai Li**

[📄 Paper](https://doi.org/10.1109/TITS.2026.3729124) · [🔍 Overview](#-overview) · [🧩 Method](#-method) · [🧪 Experiments](#-experiments) · [📝 Citation](#-citation)

</div>

---

Official implementation of **SRMARL**, published in **IEEE Transactions on Intelligent Transportation Systems (2026)**.

## 🔍 Overview

Merging safely is a team effort—especially when automated vehicles share the road with human drivers. Vehicles must coordinate with limited observations, account for rare but severe risks, and adapt to changing traffic conditions.

**SRMARL** brings these challenges into a unified safety-oriented multi-agent reinforcement learning framework. It combines physically consistent state reconstruction, tail-risk-aware policy learning, a progressively tightened safety curriculum, and operational design domain (ODD) evaluation.

> 🧠 **Reconstruct missing states** · 🛡️ **Regulate tail risks** · 📈 **Tighten constraints progressively** · 🗺️ **Map safe operating limits**

## ✨ Key Features

| Component | What it does | Why it matters |
| --- | --- | --- |
| 🧠 Physics-informed reconstruction | Recovers vehicle states using observations and kinematic consistency | Supports decisions under incomplete observations |
| 🛡️ Tail-risk modeling | Combines a Gaussian–Generalized Pareto mixture with Conditional Value at Risk (CVaR) | Accounts for rare, high-cost outcomes beyond average risk |
| 📈 Safety curriculum | Gradually raises the CVaR confidence level and lowers the safety cost limit | Introduces stricter safety requirements progressively during learning |
| 🗺️ Safety-boundary evaluation | Characterizes safe operating regions across ODD conditions | Assesses operating limits beyond aggregate performance metrics |

## 🧩 Method

### 🧠 Physics-Informed State Reconstruction

A physics-informed neural network (PINN) reconstructs vehicle states from incomplete observations. Supervised learning is combined with kinematic consistency constraints to encourage physically plausible reconstructions.

### 🛡️ Tail-Risk-Aware Constrained Learning

A Gaussian–Generalized Pareto mixture models long-term safety costs, capturing both the central distribution and its upper tail. CVaR summarizes high-cost outcomes and provides the risk measure used in constrained policy learning.

### 📈 Constraint-Homotopy Curriculum

The learning process begins with relaxed safety requirements and progressively moves toward the target constraints. The CVaR confidence level increases from **0.60 to 0.95**, while the safety cost limit decreases from **5.0 to 1.5** over **50 curriculum stages**.

### 🗺️ ODD-Based Safety-Boundary Evaluation

After policy training, ODD-based analysis identifies safe and unsafe operating regions under varying traffic conditions. This offline evaluation characterizes system-level safe operating limits and complements episode-level performance metrics.

## 🧪 Experiments

Experiments use a modified **highway-env** simulator with a kinematic bicycle model and evaluate performance across traffic densities, CAV penetration rates, road geometries, and observation conditions.

| Setting | Configuration |
| --- | --- |
| Traffic density | Low: 1–3 CAVs and 1–3 HDVs; medium: 2–4 CAVs and 2–4 HDVs; high: 4–6 CAVs and 4–6 HDVs |
| CAV penetration rates | 25%, 50%, and 75%, with 8 vehicles in total |
| Road geometries | One or two mainline lanes |
| Missing observation rates | 0%, 15%, and 75% |

The paper compares SRMARL with **MAPPO**, **MAPPO-Lag**, **MAACKTR**, **Safe-MAA2C**, **MAPPO-PIS**, and **MPC**. Evaluation metrics include success rate, crash ratio, average reward, average cost, and average velocity. A successful episode requires all ramp vehicles, including CAVs and HDVs, to merge into the main lane.

## 📊 Results at a Glance

The paper reports the following **relative improvements over MAPPO-Lag**:

| Metric | Improvement | Evaluation setting |
| --- | --- | --- |
| 🎯 Success rate | **↑ 29.17%** | Single-lane, medium-density traffic; 15% missing observations |
| 💥 Crash ratio | **↓ 77.79%** | Single-lane, medium-density traffic; 15% missing observations |
| 🗺️ Safety boundary | **↑ 63.35%** | CAV penetration rate of 75% |

See the paper for complete comparisons, ablation studies, and safety-boundary evaluation protocols.

---
## Installation
- create an python virtual environment: `conda create -n marl_cav python=3.6 -y`
- active the virtul environment: `conda activate marl_cav`
- install pytorch (torch>=1.2.0): `pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`
- install the requirements: `pip install -r requirements.txt`

## Usage
To run the code, just run it via `python run_srmarl.py`.  The config files contain the parameters for the MARL policies.


## 📝 Citation

If you use this code or find this work useful in your research, please cite:

```bibtex
@ARTICLE{Teng2026TITS,
  author={Teng, Jingjia and Li, Yang and Huang, Wenjie and Wang, Jiawei and Qin, Hongmao and Hu, Manjiang and Huang, Helai and Li, Bai},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  title={Safety-Regulated Curriculum Multi-Agent Reinforcement Learning for Cooperative On-Ramp Merging in Heterogeneous Traffic},
  year={2026},
  volume={},
  number={},
  pages={1-17},
  keywords={Costing;Costs;Safety;Modeling;Vehicles;Optimization;Learning (artificial intelligence);Merging;Modules (abstract algebra);Timing;On-ramp merging;multi-agent reinforcement learning;constrained reinforcement learning;tail risk;safety boundary;operational design domain},
  doi={10.1109/TITS.2026.3729124}
}
```
