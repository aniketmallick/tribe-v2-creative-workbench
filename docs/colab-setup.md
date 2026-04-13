# Colab Setup

## Recommended Runtime

- `Hardware accelerator`: `T4 GPU`

Use a fresh runtime. Do not debug this in a polluted notebook session.

## Install

```python
%pip install -q --upgrade pip setuptools wheel

%pip install -q --no-cache-dir \
  "torch==2.6.0" \
  "torchvision==0.21.0" \
  "torchaudio==2.6.0" \
  "numpy==2.2.6" \
  "scipy==1.15.3" \
  "scikit-learn==1.6.1" \
  "jedi>=0.16" \
  "wrapt"

%pip install -q --no-cache-dir \
  "git+https://github.com/facebookresearch/tribev2.git"

%pip install -q --no-cache-dir \
  nibabel matplotlib seaborn colorcet nilearn pyvista scikit-image huggingface_hub
```

## Restart

After install, restart the runtime once.

## Verify

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
!nvidia-smi
```

## Login

```python
from huggingface_hub import notebook_login
notebook_login()
```

## Load Model

```python
from pathlib import Path
from tribev2.demo_utils import TribeModel, download_file
from tribev2.plotting import PlotBrain

CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(exist_ok=True)

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_FOLDER)
plotter = PlotBrain(mesh="fsaverage5")
```
