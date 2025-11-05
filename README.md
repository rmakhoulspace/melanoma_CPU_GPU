 Melanoma Classification – CPU vs GPU Preprocessing (ISIC)

This project trains a binary classifier to distinguish **benign vs malignant** skin lesions using images from the **ISIC Archive**, and compares:

- A **GPU “parallel” pipeline** with a custom CUDA preprocessing kernel (via CuPy).
- A **CPU “sequential” pipeline** with naive triple-loop normalization.

> ⚠️ **Disclaimer:** This project is for **educational purposes only**.  
> It must **not** be used for medical diagnosis or any clinical decision-making.

---

## Key Ideas

- End-to-end pipeline in a **single Python script**:
  - Download ISIC data via `isic-cli`.
  - Split into **train / val / test**.
  - Parallel pipeline:
    - Custom CUDA kernel normalizes RGB images per channel using shared memory tiling.
    - CNN trained on GPU (`torch`).
  - Sequential pipeline:
    - Per-pixel normalization via triple nested loops on CPU.
    - Same CNN architecture trained on CPU.
- Both pipelines use the **same CNN** (2 conv layers + FC head) so the comparison focuses on preprocessing and device differences.

---

## Requirements

Python packages (minimal):

- `torch`, `torchvision`
- `cupy-cudaXX` (matching your CUDA version, e.g. `cupy-cuda12x`)
- `scikit-learn`
- `tqdm`
- `Pillow`
- `isic-cli`

Example install (adjust `cupy-…` as needed):

```bash
pip install torch torchvision scikit-learn tqdm pillow isic-cli cupy-cuda12x

You also need:

A GitHub/ISIC account configured for isic-cli.

A GPU with compatible CUDA drivers to fully benefit from the parallel pipeline (the script can fall back to CPU if CUDA is unavailable).

How It Works

Download data
The script uses isic-cli to fetch images from ISIC, filtering by metadata field diagnosis_1:

diagnosis_1:"Benign" → benign class

diagnosis_1:"Malignant" → malignant class

Images are stored in dataset/benign and dataset/malignant.

Split dataset
The script splits images into:

dataset/train/<class>

dataset/val/<class>

dataset/test/<class>

Parallel pipeline (GPU)

A CuPy RawKernel:

Operates on tiles (tile_size x tile_size) in shared memory.

Normalizes each RGB channel using ImageNet statistics.

The CNN is trained on GPU and evaluated on val/test.

Sequential pipeline (CPU)

Per-pixel normalization implemented as a triple nested loop.

Normalized tensors are saved and then loaded for training.

Same CNN as in the GPU pipeline, but trained on CPU.

The script prints both classification metrics and total runtime for each pipeline, so you can compare CPU vs GPU.

