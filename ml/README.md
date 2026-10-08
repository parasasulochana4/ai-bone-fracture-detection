# Fracture model training

Trains the DenseNet121 fracture classifier that the website runs in the browser
(`public/models/fracture-densenet121.onnx`). CPU-only friendly.

## Data

[`Hemg/bone-fracture-detection`](https://huggingface.co/datasets/Hemg/bone-fracture-detection)
on Hugging Face (8,863 X-rays, fractured / not fractured — a mirror of the Kaggle
"Bone Fracture Multi-Region X-ray" dataset).

**Important:** the dataset contains many augmented copies (rotations, flips, crops) of the
same X-ray. A random train/test split puts copies of test images into training and gives an
inflated ~96% accuracy. `train.py` groups near-duplicates (DenseNet embedding similarity) and
splits by group, so every copy of an X-ray is on the same side of the split. All reported
numbers use this leak-free split.

## Pipeline

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python download.py         # -> raw/<index>_<label>.png
.venv/bin/python extract.py          # cache DenseNet121 features -> gap.npy, mid.npy, labels.npy
.venv/bin/python train.py 0.95       # leak-free group split -> split.npz (+ frozen baseline)
.venv/bin/python finetune.py 8       # fine-tune denseblock4 + classifier
.venv/bin/python extract2.py         # cache transition-2 features
.venv/bin/python finetune2.py 10     # fine-tune denseblock3 + denseblock4 + classifier
.venv/bin/python compare.py          # frozen ResNet50 / EfficientNet-B0 / DenseNet121 baselines
.venv/bin/python export.py block3    # ONNX export + end-to-end test metrics -> metrics.json
```

Copy `fracture-densenet121.onnx` to `public/models/` and update
`src/lib/model-metrics.json` from `metrics.json` / `baselines.json`.
