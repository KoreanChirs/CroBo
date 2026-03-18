# pretraining

Our codebase is built on top of [ToBo](https://github.com/naver-ai/tobo) and [RSP](https://github.com/huiwon-jang/RSP). We thank the authors for their great work.

## 1. Environment Setup

Our environments are based on the `nvcr.io/nvidia/pytorch:21.10-py3` Docker image.

**Install PyTorch:**
```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

**Install packages:**
```bash
pip install timm==0.4.12
pip install decord==0.6.0
pip install tensorboardX
pip install pandas
pip install scikit-image
```

## 2. Dataset

We use Kinetics-400 for pre-training. Follow the steps below to download and prepare the dataset.

### a. Download & Extract

```bash
sh data_preprocessing/download.sh
sh data_preprocessing/extract.sh
```

We assume the root directory for the data is:
```
$DATA_ROOT = /data/kinetics400
```
If you want to use a different directory, change `root_dl` in both `download.sh` and `extract.sh`.

### b. Organize by Class

Use the provided script to organize videos into class folders:

```bash
python data_preprocessing/class_organize.py
```

> **Note:** After organizing, manually remove any special characters (e.g., spaces, parentheses) from class folder names.

### c. Resize to 224×224

```bash
python data_preprocessing/make_224scale.py
```

## 3. Pre-training

**Key files:**
- `models_crobo.py` — CroBo model architecture (encoder, decoder, loss)
- `main_pretrain_crobo.py` — training entry point (arguments, data loader, optimizer setup)
- `engine_crobo.py` — per-epoch training loop
- `util/kinetics_mfmae.py` — Kinetics-400 dataset with global/local crop augmentation
- `pretrain_crobo.sh` — training launch script

**Run:**
```bash
sh pretrain_crobo.sh
```
