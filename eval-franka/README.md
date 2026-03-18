# eval-franka

> **We strongly recommend going through `eval-DMC` first.** The DMC codebase is simpler and serves as a better starting point for understanding the overall evaluation pipeline before diving into the Franka Kitchen setup.

## 1. Codebase

This evaluation is built on top of [R3M](https://github.com/facebookresearch/r3m), a visual representation learning framework for robot manipulation. Clone the repository and follow its installation instructions before proceeding.

## 2. Preparing the Demonstration Dataset

Download the Franka Kitchen demonstration dataset from the [R3M repository](https://drive.google.com/drive/folders/1-4pOXZQ3iinbf0cLLxhZraf0Pbz4gXcG) and place it wherever you prefer on your system.

Then update the data path in `evaluation/r3meval/core/train_loop.py` — set the `data_dir` variable inside the `bc_train_loop` function to point to your dataset location.

## 3. Environment Setup

Our experiments were conducted on **Ubuntu 20.04** with an **NVIDIA RTX 4090** GPU.

We provide a `dockerfile` and `run_docker.sh` for easy environment setup. Note that you should build the Docker image for CortexBench first, which can be done in the `eval-DMC` directory.

If the Dockerfile does not work in your environment, please refer to the original [R3M](https://github.com/facebookresearch/r3m) repository for alternative setup instructions.

### GPU Rendering Check

After setting up the environment, we strongly recommend verifying that your rendering process is using the GPU. While the evaluation can run with CPU-only rendering, it is significantly slower.

To check, run `nvidia-smi` while your evaluation code is running and look at the **Type** column for your process:
- `C+G` — GPU is being used for rendering ✓
- `C` — CPU-only rendering (works, but extremely slow)

Make sure your process shows `C+G` before running full evaluations.

## 4. Evaluating a Custom Pre-trained ViT Model

The necessary files for the steps below are provided in `r3m_modify/` in our repo.

### Step 1. Add `VC1Enc` class
Inside `evaluation/r3meval/utils/obs_wrappers.py`, add the `VC1Enc` class. This class is extracted from CortexBench — refer to `r3m_modify/vc1enc.py` in our repo.

### Step 2. Add custom model loading branch
In the load-related section of `evaluation/r3meval/utils/obs_wrappers.py`, add the following `elif` block:

```python
elif "vc1" in load_path:
    from custom_utils import load_pretrained_model
    model, embedding_dim, transforms, metadata = load_pretrained_model(
        load_path=load_path,
        input_type=np.ndarray,
    )
    embedding = VC1Enc(model).eval()
    self.transforms = transforms
```

### Step 3. Add `custom_utils.py`
Copy `r3m_modify/custom_utils.py` from our repo to `evaluation/custom_utils.py` in the R3M repo.

### Step 4. Update the forward function
Add the necessary lines from `r3m_modify/forward.py` in our repo into the `forward` (encode_batch, observation) function in `evaluation/r3meval/utils/obs_wrappers.py`.

### Step 5. Add model config and checkpoint
- Place your custom checkpoint under `evaluation/custom_models/checkpoints/`
- Place the corresponding YAML config under `evaluation/custom_models/yaml/`
- **Always prefix the YAML filename with `vc1_`** (e.g., `vc1_crobo.yaml`). An example YAML is provided as `vc1_crobo.yaml` in our repo.

### Step 6. Copy vc_models from CortexBench
Copy `eai-vc/vc_models/src/vc_models` (from CortexBench) into the `evaluation/` directory of the R3M repo so that the YAML-based loading functions work correctly.

---

### Quick Reference: Adding a New Custom Model

1. Put the checkpoint at `evaluation/custom_models/checkpoints/`
2. Create the corresponding YAML at `evaluation/custom_models/yaml/` — the YAML filename must match the model name exactly (e.g., `vc1_mymodel.yaml`)
3. The checkpoint filename inside the YAML must match the actual checkpoint file
4. Write an eval script and set `num_demos=25`
5. Set `env_kwargs.load_path` to your model name — always prefix with `vc1_` when using a ViT encoder

## 5. Run Evaluation

The evaluation script is provided in our repo. Run it from the `eval-franka` directory:

```bash
bash eval_run.sh
```
