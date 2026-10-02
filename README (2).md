# Offline Vision-X Python tools

This directory contains the trainable custom detector scaffold. It is independent of the runtime/backend and uses an anchor-free grid detector implemented here (no YOLO or pretrained weights). PyTorch is required only for training/inference; image loading uses Pillow.

## Dataset format

Create `datasets/myset/classes.txt` (one class name per line) and `train.jsonl` / `val.jsonl`. Each JSONL record is:

```json
{"image":"images/frame001.jpg","width":640,"height":480,"boxes":[{"class_id":0,"x":100,"y":80,"width":60,"height":120}]}
```

Coordinates are pixel-valued top-left x/y and width/height in the original image. Image paths are relative to the JSONL file's directory. No sample data or claimed metrics are included.

Install optional dependencies with `python -m pip install torch pillow`. From this directory:

```sh
python -m visionx.training.train --train datasets/myset/train.jsonl --val datasets/myset/val.jsonl --classes datasets/myset/classes.txt --output ../models/detector.pt
python -m visionx.evaluation.evaluate --checkpoint ../models/detector.pt --annotations datasets/myset/val.jsonl --classes datasets/myset/classes.txt
```

The architecture predicts objectness, class, and normalized box offsets for one object per grid cell. This is an educational baseline architecture, not a pretrained or production detector. The matching loss is intentionally simple; evaluate and tune against a representative labeled dataset before use.
