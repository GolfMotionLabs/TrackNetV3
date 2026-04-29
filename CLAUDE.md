# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrackNetV3 is a deep learning system for **shuttlecock tracking in badminton videos**. It uses two sequential neural networks:

1. **TrackNet** — U-Net-like encoder-decoder that takes a sequence of video frames and outputs heatmaps indicating ball location
2. **InpaintNet** — 1D convolutional encoder-decoder that refines coordinates by inpainting missing/occluded trajectory points

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
# Train TrackNet (step 1)
python train.py --model_name TrackNet --seq_len 8 --epochs 30 --batch_size 10 --bg_mode concat --alpha 0.5 --save_dir exp --verbose

# Generate masks for InpaintNet training (step 2)
python generate_mask_data.py --tracknet_file ckpts/TrackNet_best.pt --batch_size 16

# Train InpaintNet (step 3)
python train.py --model_name InpaintNet --seq_len 16 --epochs 300 --batch_size 32 --lr_scheduler StepLR --mask_ratio 0.3 --save_dir exp --verbose

# Resume training
python train.py --model_name TrackNet --epochs 30 --save_dir exp --resume_training --verbose
```

### Evaluation
```bash
# Full pipeline evaluation
python test.py --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir eval

# TrackNet only
python test.py --tracknet_file ckpts/TrackNet_best.pt --save_dir eval

# With detailed prediction output for error analysis
python test.py --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir eval --output_pred
```

### Inference
```bash
# Predict from video
python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction

# With video output
python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --output_video

# Large video (streaming mode)
python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --large_video --max_sample_num 1800
```

### Data Preprocessing
```bash
python preprocess.py
python error_analysis.py --split test --host 127.0.0.1
```

## Architecture

### Data Flow

**Training pipeline:**
```
preprocess.py → [frames, median images, splits]
  → train.py (TrackNet) → TrackNet_best.pt
  → generate_mask_data.py → [mask CSVs]
  → train.py (InpaintNet) → InpaintNet_best.pt
```

**Inference pipeline:**
```
Video → frames → median background
  → Shuttlecock_Trajectory_Dataset
  → TrackNet (with temporal ensemble) → heatmaps → (x, y) coords
  → [optional] InpaintNet (with temporal ensemble) → refined coords
  → CSV output (Frame, X, Y, Visibility)
```

### Key Files

| File | Role |
|------|------|
| `model.py` | Network architectures: TrackNet, InpaintNet, and building blocks |
| `dataset.py` | `Shuttlecock_Trajectory_Dataset` and `Video_IterableDataset` |
| `train.py` | Training loop for both models |
| `test.py` | Evaluation with metrics (Accuracy, Precision, Recall, F1) |
| `predict.py` | Video inference with temporal ensemble |
| `utils/general.py` | Shared utilities: model I/O, preprocessing helpers, constants |
| `utils/metric.py` | Loss functions (WBCELoss, MSELoss) and metrics |
| `utils/visualize.py` | Heatmap overlays and trajectory plots |
| `preprocess.py` | Frame extraction, median image generation, dataset splitting |
| `generate_mask_data.py` | Creates inpainting masks from TrackNet predictions |
| `error_analysis.py` | Interactive Dash/Plotly dashboard for error analysis |

### Model Architecture Details

**TrackNet**: U-Net with skip connections. Input shape `(B, 3*seq_len, H, W)` (optionally with background channel); output `(B, seq_len, H, W)` heatmaps. Supports `bg_mode` in `{'subtract', 'subtract_concat', 'concat'}`.

**InpaintNet**: 1D temporal convolutions with skip connections. Input: coordinate sequences with inpainting masks; output: refined coordinates for masked positions.

### Important Constants (`utils/general.py`)
- `HEIGHT = 288, WIDTH = 512` — network input resolution
- `SIGMA = 2.5` — Gaussian heatmap spread
- `COOR_TH` — coordinate threshold for filtering detections

### Temporal Ensemble Modes
Used in both `test.py` and `predict.py` to average predictions across overlapping windows:
- `nonoverlap` — non-overlapping windows
- `average` — uniform weights over overlapping windows
- `weight` — center frames weighted higher than edges

### Dataset Caching
The dataset caches frames as `.npy` files to speed up data loading. **Delete these files manually when modifying dataset code** — stale caches will silently serve wrong data.

### Checkpoint Format
```python
{'epoch', 'max_val_acc', 'model', 'optimizer', 'scheduler', 'param_dict'}
```
`param_dict` contains all training hyperparameters and is used to restore configuration when resuming training.

### Prediction Types (for error analysis)
- **TP** — correct detection within threshold
- **TN** — correctly predicted no ball
- **FP1** — nearby miss (ball present but detection off)
- **FP2** — false detection (no ball but predicted one)
- **FN** — missed detection

## Requirements

- CUDA-capable GPU (code calls `.cuda()` throughout)
- Python 3.8+
- Dependencies in `requirements.txt` / `pyproject.toml`
