---
name: Bug report
about: Create a bug report to help us improve Alpamayo
title: "[BUG]"
labels: "? - Needs Triage, bug"
assignees: 'yesfandiari'

---

**Describe the bug**
A clear and concise description of what the bug is.

**Steps/Code to reproduce bug**
Follow this guide http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports to craft a minimal bug report. This helps us reproduce the issue and resolve it more quickly.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment overview (please complete the following information)**
 - Deployment: [local from source (uv), Slurm, or Cloud (specify provider)]
 - Install method: `uv venv` + `uv sync` — paste `uv --version` and Python version (3.12.x expected)
 - Model checkpoint: [e.g. nvidia/Alpamayo-R1-10B]; HuggingFace gated access granted? (yes/no)
 - Dataset: PhysicalAI-Autonomous-Vehicles access granted? (yes/no)

**Environment details**
 - Hardware: GPU type(s) and VRAM (≥24 GB required — e.g. RTX 4090, A5000, H100), number of GPUs
 - Operating System (Linux tested; others unverified)
 - CUDA / NVIDIA driver version (from `nvidia-smi`)
 - Inference settings: `num_traj_samples`, and script/notebook used (`src/alpamayo_r1/test_inference.py` or `notebook/inference.ipynb`)

**Additional context**
Add any other context about the problem here.
