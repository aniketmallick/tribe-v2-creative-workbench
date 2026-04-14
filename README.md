# TRIBE v2 Comparison Workbench

Compare how different video stimuli activate the brain using Meta's TRIBE v2.

![Demo](assets/demo.gif)
<!-- TODO: replace with an actual app capture or short screen recording. -->

## Why this exists

This project compares two video stimuli, runs TRIBE v2 on both, and visualizes each cortical prediction plus a time-varying difference map. It is intended for research, exploration, and creative-analysis workflows; it is not a medical product and must not be used for diagnosis or treatment.

## Quick Start

1. Clone the repo.

```bash
git clone https://github.com/aniketmallick/tribe-v2-creative-workbench.git
cd tribe-v2-creative-workbench
```

2. Install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is the canonical entrypoint and includes `requirements.inference.txt` + `requirements.viz.txt`.

3. Run demo mode.

```bash
python demo.py
```

## Full Usage

### Dependency profiles

- `requirements.txt`: full runtime for demo mode, app mode, inference, and visualization.
- `requirements.inference.txt`: inference stack only (TRIBE v2 + PyTorch + scientific deps).
- `requirements.viz.txt`: visualization stack only (`nilearn`, `plotly`).

### Run demo mode

`demo.py` loads cached arrays from `sample_data/` and launches the UI without live model inference.

```bash
python demo.py
```

If required files are missing, the script prints exactly which files to create.

### Run the full app

`app.py` launches the comparison UI with upload inputs and live TRIBE v2 inference.

```bash
python app.py
```

Upload two videos and click **Run Comparison**.

### Use your own videos

You can use your own `.mp4` inputs either in the app (upload both files) or from the CLI.

```bash
python compare.py /path/to/ad_a.mp4 /path/to/ad_b.mp4 --output-dir outputs
```

### Run `compare.py` directly from the CLI

`compare.py` saves aligned predictions, difference arrays, and metadata.

```bash
python compare.py /path/to/ad_a.mp4 /path/to/ad_b.mp4 \
  --output-dir outputs \
  --cache-dir ./cache \
  --model-id facebook/tribev2
```

Optional: render saved outputs as HTML brain surfaces.

```bash
python viz.py --pred-a outputs/pred_A.npy --pred-b outputs/pred_B.npy --diff outputs/diff.npy --time-step 0
```

## Repo structure

- `compare.py`: TRIBE v2 loading, per-video inference, alignment, difference computation, and output serialization.
- `viz.py`: fsaverage5 surface loading and Plotly rendering for Ad A, Ad B, and difference maps.
- `app.py`: Gradio app for uploading two videos and inspecting time-step differences interactively.
- `demo.py`: cached demo launcher that boots the app from files in `sample_data/`.
- `sample_data/`: optional lightweight demo assets (`pred_A.npy`, `pred_B.npy`, `diff.npy`, optional metadata/videos).

## How it works

- TRIBE v2 converts stimuli into cortical predictions on the fsaverage5 surface.
- This project aligns two prediction sequences and computes a time-varying difference map.
- The UI lets users inspect those differences interactively.

## License

This project's code is licensed under MIT.
TRIBE v2 model weights are property of Meta and licensed under CC-BY-NC-4.0.
This project does not distribute model weights — users must fetch them from
HuggingFace (facebook/tribev2). The text encoder path requires gated access
to Llama 3.2-3B.

## Acknowledgments

- [TRIBE v2 GitHub repository](https://github.com/facebookresearch/tribev2)
- [Hugging Face model card](https://huggingface.co/facebook/tribev2)
- [Meta demo page](https://aidemos.atmeta.com/tribev2)
- [Paper: A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/)
