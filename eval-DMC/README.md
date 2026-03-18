# eval-DMC

## 1. Codebase

This evaluation is built on top of [CortexBench](https://github.com/facebookresearch/eai-vc), a suite of robotics evaluation environments for visual representation learning. CortexBench includes several benchmarks across embodied AI tasks; among them, we use the **DeepMind Control Suite (DMC)** environment for our experiments. Clone the repository and follow its installation instructions before proceeding.

## 2. Preparing the Demonstration Dataset

Create the dataset directory and download the DMC expert demonstrations:

```bash
mkdir -p cortexbench/mujoco_vc/visual_imitation/data/datasets
cd cortexbench/mujoco_vc/visual_imitation/data/datasets

wget https://dl.fbaipublicfiles.com/eai-vc/mujoco_vil_datasets/dmc-expert-v1.0.zip
unzip dmc-expert-v1.0.zip
rm dmc-expert-v1.0.zip
```

## 3. Environment Setup

Our experiments were conducted on **Ubuntu 20.04** with an **NVIDIA RTX 4090** GPU.

We provide a `dockerfile` and `run_docker.sh` for easy environment setup. If the Dockerfile does not work in your environment, please refer to the original [eai-vc](https://github.com/facebookresearch/eai-vc) repository for alternative setup instructions.

### GPU Rendering Check

After setting up the environment, we strongly recommend verifying that your rendering process is using the GPU. While the evaluation can run with CPU-only rendering, it is significantly slower.

To check, run `nvidia-smi` while your evaluation code is running and look at the **Type** column for your process:
- `C+G` — GPU is being used for rendering ✓
- `C` — CPU-only rendering (works, but extremely slow)

Make sure your process shows `C+G` before running full evaluations.

## 4. Evaluating a Custom Pre-trained ViT Model

### Step 1. Create a custom model config YAML

Create a YAML file at:
```
vc_models/src/vc_models/conf/model/custom_model.yaml
```
This file specifies how to load your checkpoint. The loading functions are primarily implemented in `vc_models/src/vc_models/models/vit/vit.py`.

- **If you use a ViT encoder:** modify the `load_mae_encoder` function in `vit.py` by adding `strict=False` to `model.load_state_dict(...)`. This is necessary because our decoder architecture differs from the original, but since the decoder is not used at evaluation time, `strict=False` is completely safe.
- **If you use a different architecture:** implement your own loading function following the same output format as the existing load functions in `vit.py`.

### Step 2. Place your checkpoint

Put your model checkpoint under:
```
vc_models/src/model_ckpts/
```

That's all the setup needed to plug in a custom model for evaluation.

## 5. Run Evaluation

```bash
cd cortexbench/mujoco_vc/visual_imitation
bash eval_run.sh
```
