# AStar Perception Pipeline

## Overview

This project is a modular perception pipeline for detecting, tracking, and describing people from images, videos, and webcam input.

Current pipeline:

`image / video / webcam -> YOLO -> ByteTrack -> stable person ID -> crop -> VLM`

Each detected person can have:

- bounding box
- persistent logical ID
- semantic description

The system supports:

- still-image inference
- recorded video processing
- live webcam processing
- asynchronous VLM inference for webcam mode
- appearance-based ID re-matching after short disappear / reappear events
- automatic stale-track deletion after a timeout

## Project Structure

- `main.py`
  CLI entrypoint for image, video, webcam, or camera-index input.
- `pipeline/pipeline.py`
  Main orchestration layer for detection, tracking, rematching, cropping, async VLM jobs, and output formatting.
- `detection/yolo_detector.py`
  YOLO + ByteTrack wrapper with automatic GPU selection when CUDA PyTorch is available.
- `vlm/vlm.py`
  Vision-language client for Ollama-compatible chat completions and structured response parsing.

## Core Features

- YOLO person detection
- ByteTrack multi-object tracking
- stable logical person IDs above raw ByteTrack IDs
- appearance-based re-matching using lightweight HSV embeddings
- per-person crop extraction
- VLM-generated structured person descriptions
- async webcam VLM worker so the live camera window does not freeze
- GPU-backed YOLO/ByteTrack when CUDA-enabled PyTorch is installed

## Identity Handling

The project uses two layers of identity:

1. ByteTrack ID
   This is the raw short-term tracking ID from the tracker.
2. Person ID
   This is the stable logical ID shown by the pipeline.

If a person disappears and reappears within the rematch window, the system tries to attach the new ByteTrack ID back to the old logical person ID using appearance similarity.

By default, stale identities are deleted after `5` seconds if nothing matches them.

## Installation

### 1. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Make sure Ollama is running

The project expects an Ollama-compatible endpoint at:

```text
http://localhost:11434/v1/chat/completions
```

## Running the Project

### Image

```powershell
python .\main.py .\test_image.jpg
```

### Video

```powershell
python .\main.py .\two_people.mp4
```

### Webcam

```powershell
python .\main.py webcam
```

You can also choose a camera index directly:

```powershell
python .\main.py 0
python .\main.py 1
```

Press `q` to quit webcam mode.

## Environment Variables

The current pipeline exposes several runtime knobs through environment variables:

### VLM settings

- `VLM_MODEL`
  Ollama model name. Default: `qwen2.5vl:3b`
- `VLM_URL`
  VLM endpoint URL. Default: `http://localhost:11434/v1/chat/completions`
- `VLM_TIMEOUT_SEC`
  Request timeout in seconds. Default: `120`
- `VLM_MAX_CROP_SIDE`
  Maximum crop size sent to the VLM. Default: `896`
- `VLM_QUEUE_SIZE`
  Max pending webcam VLM jobs. Default: `8`

### Identity / rematch settings

- `PERSON_REMATCH_TIMEOUT_SEC`
  How long to keep unmatched identities alive. Default: `5`
- `PERSON_REMATCH_THRESHOLD`
  Appearance similarity threshold for re-matching. Default: `0.82`

Example:

```powershell
$env:VLM_MODEL="qwen2.5vl:3b"
$env:PERSON_REMATCH_TIMEOUT_SEC="7"
$env:PERSON_REMATCH_THRESHOLD="0.85"
python .\main.py webcam
```

## GPU Notes

### YOLO / ByteTrack

YOLO and ByteTrack will use the GPU automatically if CUDA-enabled PyTorch is available in the project venv.

This project has already been updated to prefer:

```text
cuda:0
```

when `torch.cuda.is_available()` is true.

### Ollama / VLM

Ollama GPU use depends on whether the selected vision model fits your available VRAM.

Important limitation:

- on a 6 GB GPU, `qwen2.5vl:3b` may still fall back to CPU
- if Ollama shows `100% CPU` in `ollama ps`, the model is not being offloaded to GPU

Useful checks:

```powershell
ollama ps
nvidia-smi
```

If the current Qwen vision model does not fit, you may need to switch to a smaller Ollama vision model via `VLM_MODEL`.

## Output Behavior

### Image mode

Returns:

- YOLO detections
- per-person VLM descriptions

### Video mode

Returns frame-by-frame results with:

- stable person ID
- bbox
- confidence
- class
- description

### Webcam mode

Shows a live OpenCV window with:

- tracked boxes
- stable person IDs
- semantic action labels

If the VLM is still working on a person, the overlay temporarily shows:

```text
Analyzing...
```

## Current Limitations

- Webcam mode is asynchronous, but recorded video mode is still synchronous for VLM inference.
- Re-matching is based on lightweight HSV appearance similarity, not a dedicated re-identification model.
- Ollama GPU usage is model- and VRAM-dependent and cannot be forced from this Python code if the model does not fit.
- Structured JSON quality depends on the selected VLM and prompt adherence.

## Quick Validation

You can run a syntax check with:

```powershell
python -m py_compile .\main.py .\pipeline\pipeline.py .\detection\yolo_detector.py .\vlm\vlm.py
```

## Summary

This project is no longer just a basic YOLO + VLM image demo. It is now a real-time person perception pipeline with:

- tracking
- stable identity handling
- re-matching
- async webcam inference
- GPU-enabled detection/tracking
- configurable local VLM integration
