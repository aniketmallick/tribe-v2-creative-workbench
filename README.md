# TRIBE v2 Creative Workbench

Open-source starter repo for experimenting with Meta's `TRIBE v2` model on ad creatives, videos, audio, and text.

## What This Repo Is

This repo is a clean public starting point for:

- running `TRIBE v2` in Colab or Kaggle
- comparing two creative variants
- turning raw cortical predictions into something usable for product experiments

## What TRIBE v2 Actually Does

`TRIBE v2` predicts average-subject fMRI-like cortical activity from `video`, `audio`, or `text` inputs.

This is useful for:

- comparing two ad cuts
- testing narration vs no narration
- testing subtitle variants
- exploring which modalities dominate predicted cortical response over time

## Current Scope

This repo currently contains:

- setup instructions for `Colab` and `Kaggle`
- a stable dependency set for notebook-based inference
- utility scripts for environment checks
- a public repo structure you can build on

## Project Roadmap

1. Get stable inference running in Colab and Kaggle.
2. Add a comparison notebook for `creative A vs creative B`.
3. Build a simple app viewer with synced brain activity and stimulus playback.
4. Add ROI summaries and difference views.

## Quick Start

### Kaggle

1. Create a notebook.
2. Enable `Internet`.
3. Use `GPU T4 x2`, not `P100`.
4. Add a Kaggle secret named `HF_TOKEN`.
5. Follow [docs/kaggle-setup.md](docs/kaggle-setup.md).

### Colab

1. Open a fresh notebook.
2. Switch to a `T4 GPU`.
3. Make sure your Hugging Face token is ready.
4. Follow [docs/colab-setup.md](docs/colab-setup.md).

## License Boundary

The code in this repo is released under `MIT`.

Important:

- `TRIBE v2` model weights are not owned by this repo.
- The released `facebook/tribev2` checkpoint is licensed separately.
- At the time of writing, the model card lists `CC-BY-NC-4.0`.

Do not pretend this repo grants commercial rights to the model weights. It does not.

Official model references:

- [TRIBE v2 GitHub](https://github.com/facebookresearch/tribev2)
- [TRIBE v2 Hugging Face](https://huggingface.co/facebook/tribev2)
- [Meta demo page](https://aidemos.atmeta.com/tribev2)

## Next Build Targets

- `notebooks/creative_compare.ipynb`
- `app/` interactive viewer
- ROI-level summaries
- exportable comparison reports
