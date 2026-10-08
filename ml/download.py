# Download the dataset to raw/<index>_<label>.png (label 0 = fractured, 1 = not fractured).
import os
from datasets import load_dataset
ds = load_dataset("Hemg/bone-fracture-detection", split="train")
os.makedirs("raw", exist_ok=True)
for i, ex in enumerate(ds):
    ex["image"].convert("L").save(f"raw/{i:05d}_{ex['label']}.png")
