# Model Overview

## Model NSpect ID

NSPECT-IUM0-0RU4 (https://nvbugspro.nvidia.com/bug/5657854)
TODO: delete it before release

### API EndPoint

https://huggingface.co/nvidia/alpamayo-r1-32b (Hugging Face model repository)
TODO: delete it before release

### Description:

Alpamayo-R1 integrates Chain-of-Causation reasoning with trajectory planning to enhance decision-making in complex autonomous-driving scenarios. Alpamayo-R1 (v1.0) was developed by NVIDIA as a vision-language-action (VLA) model that bridges interpretable reasoning with precise vehicle control for autonomous-driving applications.

This model is ready for non-commercial use.

### License/Terms of Use:

[Visit the NVIDIA Legal Release Process](https://nvidia.sharepoint.com/sites/ProductLegalSupport) for instructions on getting legal support for a license selection: <br>

- If you are releasing under an open source license (such as Apache 2.0, MIT), contact the [Open Source Review Board](https://confluence.nvidia.com/pages/viewpage.action?pageId=800720661) (formerly SWIPAT) by filing a contribution bug request [here](https://nvbugspro.nvidia.com/bug/2885991). <br>

- If your release is for non-commercial or research purposes only, file a new bug [here](https://nvbugspro.nvidia.com/bug/3508089). <br>

- If your release allows for commercial purposes, submit [Product Legal Support Form](https://forms.office.com/pages/responsepage.aspx?id=FT0IQ3NywUC32znv2czBejILt4CYhTJKv0O6I4gccylUMVlMSE4xSFhYMUYyT1VMNVNCREk4RlE1NS4u&route=shorturl). <br>

Legal tracking: https://nvbugspro.nvidia.com/bug/5657854
Legal tracker sheet: https://docs.google.com/spreadsheets/d/1Gd_R2982-3GWfzTVyE0xG3evO0rmoKgxPZ_-vd7Z-AM/edit?usp=sharing

Apache 2.0

### Deployment Geography:

Global

### Use Case:

Researchers and autonomous-driving practitioners who are developing and evaluating VLA models for autonomous-driving scenarios, particularly for handling safety-critical, long-tail events.

### Release Date:

Hugging Face 12/02/2025 via https://huggingface.co/nvidia/alpamayo-r1-32b

### Inference Code:

GitHub: https://github.com/nvidia/alpamayo

## References(s):

arXiv:2511.00088v1 - Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail

## Model Architecture:

**Architecture Type:** Transformer

**Network Architecture:** A VLA model based on Cosmos-Reason and featuring a diffusion-based trajectory decoder.

**This model was developed based on:** Cosmos-Reason (VLM backbone) with a diffusion-based action decoder

**Number of model parameters:** 3.2 x 10^10

## Computational Load (Internal Only: For NVIDIA Models Only)

**Cumulative Compute:** 3.7 x 10^23 FLOPS (512 H100 GPUs for 2 weeks)
TODO: remove it before release.

**Estimated Energy and Emissions for Model Training:** Approximately 55.4 t CO₂e (based on 512 H100 GPUs for 2 weeks)
TODO: remove it before release.

## Input(s):

**Input Type(s):** Image, Text, Trajectory

**Input Format(s):**

- Image: Red, Green, Blue (RGB)
- Text: String
- Trajectory: Floating-point values (position, velocity, acceleration, steering)

**Input Parameters:**

- Image: Two-Dimensional (2D), multi-camera, multi-timestep
- Text: One-Dimensional (1D)
- Trajectory: One-Dimensional (1D)

**Other Properties Related to Input:**
Multi-camera images (2-6 cameras) with 2-second history window at 10Hz, image resolution 320x512 pixels typical. Text inputs include optional user commands and navigation instructions. Egomotion history (position, velocity, acceleration, steering) also provided.

## Output(s)

**Output Type(s):** Text, Trajectory

**Output Format(s):**

- Text: String (Chain of Causation reasoning traces)
- Trajectory: Floating-point values (x, y, yaw)

**Output Parameters:**

- Text: One-Dimensional (1D)
- Trajectory: Two-Dimensional (2D) in bird's-eye-view space

**Other Properties Related to Output:**
Outputs 6-second future trajectory (64 waypoints at 10Hz) with position (x, y) and heading (yaw) in ego vehicle coordinate frame. Text reasoning traces are variable length describing driving decisions and causal factors.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:

**Runtime Engine(s):**

- PyTorch (minimum version: 2.8)
- Hugging Face Transformers (minimum version: 4.57.1)
- DeepSpeed (minimum version: 0.17.4)

**Supported Hardware Microarchitecture Compatibility:**

- NVIDIA GPUs with sufficient memory to load 32B parameter model (minimum 2 GPUs with 40GB+ VRAM each, or 1 GPU with 80GB+ VRAM)

**Preferred/Supported Operating System(s):**

- Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):

Alpamayo-R1 32B v1.0 trained

Can be integrated into autonomous driving software in the cloud for advanced end-to-end perception, reasoning, and motion planning.

## Training, Testing, and Evaluation Datasets:

## Training Dataset:

**Link:** Chain of Causation (CoC) Dataset, Cosmos-Reason Physical AI datasets, NVIDIA's internal proprietary autonomous driving data.

**Data Modality:**

- Image (multi-camera)
- Text (reasoning traces)
- Other: Trajectory data (egomotion, future waypoints)

**Image Training Data Size:** More than 1 Billion Images (from 80,000 hours of multi-camera driving data)

**Text Training Data Size:** Less than a Billion Tokens (700K CoC reasoning traces plus Cosmos-Reason training data)

**Video Training Data Size:** 10,000 to 1 Million Hours (80,000 hours)

**Non-Audio, Image, Text Training Data Size:** Trajectory data: 80,000 hours at 10Hz sampling rate

**Data Collection Method by dataset:** Hybrid: Automatic/Sensors (camera and vehicle sensors), Synthetic (VLM-generated reasoning)

**Labeling Method by dataset:** Hybrid: Human (structured CoC annotations), Automated (VLM-based auto-labeling), Automatic/Sensors (trajectory and egomotion)

**Properties:**
The dataset comprises 80,000 hours of multi-camera driving video with corresponding egomotion and trajectory annotations.
It includes 700,000 Chain-of-Causation (CoC) reasoning traces that provide decision-grounded, causally linked explanations of driving behaviors.
Content includes machine-generated data from vehicle sensors (cameras, IMUs, and GPS) and synthetic reasoning traces.
CoC annotations are in English and use a structured format that links driving decisions to causal factors.
Sensors include RGB cameras (2-6 per vehicle), inertial measurement units, and GPS.

**Dataset License(s):** https://docs.google.com/spreadsheets/d/1Gd_R2982-3GWfzTVyE0xG3evO0rmoKgxPZ_-vd7Z-AM/edit?usp=sharing
TODO: Internal only; reminder not to publish.

### Testing Dataset:

**Link:** Proprietary autonomous driving test datasets, closed-loop simulation, on-vehicle road tests.

**Data Collection Method by dataset:** Hybrid: Automatic/Sensors (real-world driving data), Synthetic (simulation scenarios)

**Labeling Method by dataset:** Hybrid: Automatic/Sensors, Human (ground truth verification)

**Properties:**
This dataset covers multi-camera driving scenarios with a particular focus on safety-critical, long-tail events. It includes challenging cases such as complex intersections, cut-ins, pedestrian interactions, and adverse weather conditions. Data are collected from RGB cameras and vehicle sensors.

**Dataset License(s):** https://docs.google.com/spreadsheets/d/1Gd_R2982-3GWfzTVyE0xG3evO0rmoKgxPZ_-vd7Z-AM/edit?usp=sharing
TODO: Internal only; reminder not to publish.

### Evaluation Dataset:

**Link:** Same as Testing Dataset.

**Benchmark Score:**
Up to 12% improvement in planning accuracy on challenging cases vs. trajectory-only baseline (minADE: 0.994 → 0.868). 35% reduction in off-road rate and 25% reduction in close encounter rate in closed-loop simulation. RL post-training improves reasoning quality by 45% and reasoning-action consistency by 37%.
TODO: update the number when have 32B model. Reminder not to publish.

**Data Collection Method by dataset:** Hybrid: Automatic/Sensors (real-world driving data), Synthetic (simulation scenarios)

**Labeling Method by dataset:** Hybrid: Automatic/Sensors, Human (ground truth verification)

**Properties:**
Evaluation focuses on safety-critical, long-tail scenarios, including complex intersections, pedestrian crossings, vehicle cut-ins, and challenging weather and lighting conditions. Multi-camera sensor data are collected from RGB cameras.

**Dataset License(s):** https://docs.google.com/spreadsheets/d/1Gd_R2982-3GWfzTVyE0xG3evO0rmoKgxPZ_-vd7Z-AM/edit?usp=sharing
TODO: Reminder not to publish.

**Quantitative Evaluation Benchmarks:**
Planning accuracy improvement: +12% on challenging cases (minADE: 0.994 → 0.868). Off-road rate reduction: 35%. Close encounter rate reduction: 25%. Reasoning quality improvement: +45% (via RL post-training). Reasoning-action consistency improvement: +37%.
TODO: update the number when have 32B model. Reminder to remove.

# Inference:

**Acceleration Engine:** PyTorch, Hugging Face Transformers

**Test Hardware:**

- Minimum: 2 GPUs with 40GB+ VRAM each (e.g., NVIDIA A100), or 1 GPU with 80GB+ VRAM (e.g., NVIDIA H100 80GB)
- Tested on: NVIDIA H100

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
