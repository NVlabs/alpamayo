# Alpamayo-R1

[**Code**](https://github.com/NVlabs/alpamayo) | [**Paper**](https://arxiv.org/abs/2511.00088)

## Model Overview

### Description:

Alpamayo-R1 integrates Chain-of-Causation reasoning with trajectory planning to enhance decision-making in complex autonomous-driving scenarios. Alpamayo-R1 (v1.0) was developed by NVIDIA as a vision-language-action (VLA) model that bridges interpretable reasoning with precise vehicle control for autonomous-driving applications.

This model is ready for non-commercial use.

### License:

The model weights are released under a [non-commercial license](./LICENSE).

The inference code is released under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.

### Deployment Geography:

Global

### Use Case:

Researchers and autonomous-driving practitioners who are developing and evaluating VLA models for autonomous-driving scenarios, particularly for handling rare, long-tail events.

### Release Date:

Hugging Face 12/03/2025 via https://huggingface.co/nvidia/Alpamayo-R1-10B

### Inference Code:

GitHub: https://github.com/NVlabs/alpamayo

## Reference:

[Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail](https://arxiv.org/abs/2511.00088)

## Model Architecture:

**Architecture Type:** Transformer

**Network Architecture:** A VLA model based on Cosmos-Reason and featuring a diffusion-based trajectory decoder.

**This model was developed based on:** Cosmos-Reason (VLM backbone) with a diffusion-based action decoder

**Number of model parameters:**

- Backbone: 8.2B parameters
- Action Expert: 2.3B parameters

## Input(s):

**Input Type(s):** Image/Video, Text, Egomotion History

**Input Format(s):**

- Image: Red, Green, Blue (RGB)
- Text: String
- Egomotion History: Floating-point values `(x, y, z), R_rot`

**Input Parameters:**

- Image: Two-dimensional (2D), multi-camera, multi-timestep
- Text: One-dimensional (1D)
- Egomotion History: Three-dimensional (3D) translation and nine-dimensional (9D, 3x3) rotation, multi-timestep

**Other Properties Related to Input:**
Multi-camera images (4 cameras: front-wide, front-tele, cross-left, cross-right) with 0.4 second history window at 10Hz (4 frames per camera), image resolution 1080x1920 pixels (processor will downsample them to 320x576 pixels). Text inputs include user commands. Images and egomotion history (16 waypoints at 10Hz) also require associated timestamps.
Note that the model is primarily trained and only tested under this setting.

## Output(s)

**Output Type(s):** Text, Trajectory

**Output Format(s):**

- Text: String (Chain-of-Causation reasoning traces)
- Trajectory: Floating-point values `(x, y, z), R_rot`

**Output Parameters:**

- Text: One-dimensional (1D)
- Trajectory: Three-dimensional (3D) translation and nine-dimensional (9D, 3x3) rotation, multi-timestep

**Other Properties Related to Output:**
Outputs 6.4-second future trajectory (64 waypoints at 10Hz) with position `(x, y, z)` and rotation matrix `R_rot` in ego vehicle coordinate frame.
Internally, the trajectory is represented as a sequence of dynamic actions (acceleration and curvature) following a unicycle model in bird's-eye-view (BEV) space.
Text reasoning traces are variable in length, describing driving decisions and causal factors.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:

**Runtime Engine(s):**

- PyTorch (minimum version: 2.8)
- Hugging Face Transformers (minimum version: 4.57.1)
- DeepSpeed (minimum version: 0.17.4)

**Supported Hardware Microarchitecture Compatibility:**

- NVIDIA GPUs with sufficient memory to load a 10B parameter model (minimum 1 GPU with at least 24GB of VRAM)

**Preferred/Supported Operating System(s):**

- Linux (we have not tested on other operating systems)

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):

Alpamayo-R1 10B v1.0 trained

Can be integrated into autonomous driving software in the cloud for advanced end-to-end perception, reasoning, and motion planning.

## Training, Testing, and Evaluation Datasets:

## Training Dataset:

Alpamayo-R1's training data comprises a mix of Chain of Causation (CoC) reasoning traces, Cosmos-Reason Physical AI datasets, and NVIDIA's internal proprietary autonomous driving data.

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
The dataset comprises 80,000 hours of multi-camera driving videos with corresponding egomotion and trajectory annotations.
It includes 700,000 Chain-of-Causation (CoC) reasoning traces that provide decision-grounded, causally linked explanations of driving behaviors.
Content includes machine-generated data from vehicle sensors (cameras, IMUs, and GPS) and synthetic reasoning traces.
CoC annotations are in English and use a structured format that links driving decisions to causal factors.
Sensors include RGB cameras (2-6 per vehicle), inertial measurement units, and GPS.

### Testing Dataset:

**Link:** Proprietary autonomous driving test datasets, closed-loop simulation, on-vehicle road tests.

**Data Collection Method by dataset:** Hybrid: Automatic/Sensors (real-world driving data), Synthetic (simulation scenarios)

**Labeling Method by dataset:** Hybrid: Automatic/Sensors, Human (ground truth verification)

**Properties:**
This dataset covers multi-camera driving scenarios with a particular focus on rare, long-tail events. It includes challenging cases such as complex intersections, cut-ins, pedestrian interactions, and adverse weather conditions. Data are collected from RGB cameras and vehicle sensors.

### Evaluation Dataset:

**Link:** Same as Testing Dataset.

**Data Collection Method by dataset:** Hybrid: Automatic/Sensors (real-world driving data), Synthetic (simulation scenarios)

**Labeling Method by dataset:** Hybrid: Automatic/Sensors, Human (ground truth verification)

**Properties:**
Evaluation focuses on rare, long-tail scenarios, including complex intersections, pedestrian crossings, vehicle cut-ins, and challenging weather and lighting conditions. Multi-camera sensor data are collected from RGB cameras.

**Quantitative Evaluation Benchmarks:**

- Closed-Loop Evaluation using [AlpaSim](https://github.com/NVlabs/alpasim) on the [PhysicalAI-AV-NuRec Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec): AlpaSim Score of 0.72.
- Open-Loop Evaluation on the [PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles): minADE_6 at 6.4s of 0.85m.

# Inference:

**Acceleration Engine:** PyTorch, Hugging Face Transformers

**Test Hardware:**

- Minimum: 1 GPU with 24GB+ VRAM (e.g., NVIDIA RTX 3090, RTX 3090 Ti, RTX 4090, A5000, or equivalent)
- Tested on: NVIDIA H100

For scripts related to model inference, please check out our [code repository](https://github.com/NVlabs/alpamayo).

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
