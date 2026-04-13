# Kaggle Setup

## Recommended Notebook Settings

- `Accelerator`: `GPU T4 x2`
- `Internet`: `On`
- `Persistence`: `Off` for initial validation

Do not use `P100` for this workflow if you want the notebook's default `whisperx` path to work cleanly. `P100` is the wrong GPU for the float16 transcription path used here.

## Secrets

Add a Kaggle secret named `HF_TOKEN`.

## Install

```python
%pip install -q --upgrade pip setuptools wheel
%pip install -q --no-cache-dir -r requirements.inference.txt
# Optional: visualization stack for Prompt 2 (`viz.py`)
%pip install -q --no-cache-dir -r requirements.viz.txt
```

If `requirements.inference.txt` is not present inside the notebook environment, paste the requirements directly from the repo.

## Restart

After install, use `Restart & clear cell outputs`.

## Verify

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
```

Expected result:

- `True`
- a `T4` device name

## Login

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
login(user_secrets.get_secret("HF_TOKEN"))
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

## Known Failure Modes

- `401/403`: gated Hugging Face dependency not approved yet
- `ModuleNotFoundError: tribev2`: install did not land in the active kernel
- `float16 compute type`: wrong GPU or no GPU
