# RealEngine: Simulating Autonomous Driving in Realistic Context

### [[Project]]() [[Paper]]() 

> [**RealEngine: Simulating Autonomous Driving in Realistic Context**](),            
> [Junzhe Jiang](https://scholar.google.com/citations?user=gnDoDP4AAAAJ), [Nan Song](https://scholar.google.com/citations?user=wLZVtjEAAAAJ), [Jingyu Li](https://github.com/Whale-ice), [Xiatian Zhu](https://xiatian-zhu.github.io/), [Li Zhang](https://lzrobots.github.io)       

**Official implementation of "RealEngine: Simulating Autonomous Driving in Realistic Context".** 

## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>

## ▶️ Get started
### 📦 Environment
1. Prepare the environment.
```
# Clone the repo.
git clone https://github.com/fudan-zvg/RealEngine.git
cd RealEngine

# Make a conda environment.
conda create --name realengine python=3.9
conda activate realengine

# Install PyTorch according to your CUDA version
# CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Install nuplan devkits
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .

# Install raytracing
git clone https://github.com/ashawkey/raytracing
cd raytracing
pip install .
```
2. Install and download Navsim-mini as [Navsim install](https://github.com/autonomousvision/navsim/blob/main/docs/install.md), and download RealEngine scene checkpoints in [HuggingFace RealEngine](https://huggingface.co/datasets/selfspin/RealEngine/tree/main/dataset/realengine).
The folder tree is as follows:
```
dataset
├── openscene
│   ├── maps
│   └── openscene-v1.1
│       ├── navsim_logs
│       └── sensor_blobs
└── realengine
    ├── background
    │   ├── cam
    │   └── lidar
    ├── irrmaps
    ├── relighting
    └── vehicles
```
Then, build cache for navsim-mini
```
chmod +x scripts/evaluation/run_metric_caching.sh
./scripts/evaluation/run_metric_caching.sh
```

3. For the [DriveX](submodules/DriveX) and [GSLiDAR](submodules/GSLiDAR) submodules, please refer to their respective `README.md` files for installation instructions.

4. Download navsim AD agent checkpoints to `./model`.

| Agent          | Checkpoint                                                   |
| -------------- | ------------------------------------------------------------ |
| TransFuser     | [transfuser_seed_0.ckpt](https://huggingface.co/autonomousvision/navsim_baselines/tree/main/transfuser) |
| VAD            | [vad_epoch_99.ckpt](https://huggingface.co/datasets/selfspin/RealEngine/tree/main/model) |
| DiffusionDrive | [diffusiondrive_navsim_88p1_PDMS.pth](https://huggingface.co/hustvl/DiffusionDrive/tree/main) |


The folder tree is as follows:
```
model
├── diffusiondrive_navsim_88p1_PDMS.pth
├── kmeans_navsim_traj_20.npy
├── transfuser_seed_0.ckpt
└── vad_epoch_99.ckpt
```

5. Due to the complexity of the environment setup, we provide the [final pip list](docs/final_pip_list.md) of our environment to facilitate verification and reproducibility.
### 🚗 Evaluating
You can use the following command to simulating autonomous driving in realistic context.
```
# DiffusionDrive
# Non-reactive simulation.
CUDA_VISIBLE_DEVICES=0 python navsim/planning/script/run_pdm_score_with_render_base.py \
train_test_split=mini agent=diffusiondrive_agent worker=single_machine_thread_pool \
agent.checkpoint_path=model/diffusiondrive_navsim_88p1_PDMS.pth \
experiment_name=diffusiondrive_agent_eval

# Safety test simulation.
CUDA_VISIBLE_DEVICES=1 python navsim/planning/script/run_pdm_score_with_render_edit.py \
train_test_split=mini agent=diffusiondrive_agent worker=single_machine_thread_pool \
agent.checkpoint_path=model/diffusiondrive_navsim_88p1_PDMS.pth \
experiment_name=diffusiondrive_agent_eval

# Multi-agent interaction simulation.
CUDA_VISIBLE_DEVICES=1 python navsim/planning/script/run_pdm_score_with_render_multi_agent.py \
train_test_split=mini agent=diffusiondrive_agent worker=single_machine_thread_pool \
agent.checkpoint_path=model/diffusiondrive_navsim_88p1_PDMS.pth \
experiment_name=diffusiondrive_agent_eval
```
