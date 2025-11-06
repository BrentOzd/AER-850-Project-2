import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# --- setup ---
IMAGE_SIZE = (500, 500)
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
summary_path = ARTIFACTS / "training_summary.json"
if not summary_path.exists():
    raise FileNotFoundError("training_summary.json not found. Run training first.")

print("Step 5 - Test")
with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

# label maps 
idx_to_label = {int(k): v for k, v in summary["class_indices"]["index_to_label"].items()}
label_order = [idx_to_label[i] for i in sorted(idx_to_label.keys())]

# paths to both variants
model_path_A = Path(summary["comparison_table"]["variant_a"]["saved_model"]).resolve()
model_path_B = Path(summary["comparison_table"]["variant_b"]["saved_model"]).resolve()
if not model_path_A.exists(): raise FileNotFoundError(f"Variant A model missing: {model_path_A}")
if not model_path_B.exists(): raise FileNotFoundError(f"Variant B model missing: {model_path_B}")

print(f"Loading Variant B: {model_path_B}")
model_B = tf.keras.models.load_model(model_path_B)
print(f"Loading Variant A: {model_path_A}")
model_A = tf.keras.models.load_model(model_path_A)

# test set
test_images = [
    ("Project 2 Data/Data/test/crack/test_crack.jpg", "test_crack.jpg"),
    ("Project 2 Data/Data/test/missing-head/test_missinghead.jpg", "test_missinghead.jpg"),
    ("Project 2 Data/Data/test/paint-off/test_paintoff.jpg", "test_paintoff.jpg"),
]

def load_img_to_batch(p):
    pil = tf.keras.utils.load_img(p, target_size=IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(pil)
    return arr.astype(np.uint8), (arr / 255.0)[None, ...]

def predict_all(model, images):
    out = []
    for rel, disp in images:
        img_path = BASE / rel
        if not img_path.exists():
            raise FileNotFoundError(f"Missing test image: {img_path}")
        img_u8, x = load_img_to_batch(img_path)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        out.append({
            "display": disp,
            "image_u8": img_u8,
            "pred_label": idx_to_label[idx],
            "pred_prob": float(probs[idx]),
            "probs": {label_order[i]: float(probs[i]) for i in range(len(label_order))}
        })
    return out

def pretty_probs(pdict):
    return ", ".join([f"{k} {100*v:.1f}%" for k, v in pdict.items()])

# run both variants
results_B = predict_all(model_B, test_images)
results_A = predict_all(model_A, test_images)

# console readout
print("\nPredictions (full class probabilities):")
for rB, rA in zip(results_B, results_A):
    print(f"{rB['display']}:  "
          f"Variant B -> {rB['pred_label']} ({rB['pred_prob']*100:.1f}%) [{pretty_probs(rB['probs'])}]  |  "
          f"Variant A -> {rA['pred_label']} ({rA['pred_prob']*100:.1f}%) [{pretty_probs(rA['probs'])}]")

# figure for Variant B (3x1 vertical)
figB, axesB = plt.subplots(3, 1, figsize=(6, 15))
for j, r in enumerate(results_B):
    ax = axesB[j]
    ax.imshow(r["image_u8"]); ax.axis("off")
    ax.set_title(f"{r['pred_label']} ({r['pred_prob']*100:.1f}%)", fontsize=12)
figB.suptitle("Variant B — predictions on test images", fontsize=14)
figB.tight_layout(rect=[0, 0, 1, 0.96])
outB = ARTIFACTS / "test_predictions_variant_b_3x1.png"
figB.savefig(outB, dpi=150); plt.close(figB)

# figure for Variant A (3x1 vertical)
figA, axesA = plt.subplots(3, 1, figsize=(6, 15))
for j, r in enumerate(results_A):
    ax = axesA[j]
    ax.imshow(r["image_u8"]); ax.axis("off")
    ax.set_title(f"{r['pred_label']} ({r['pred_prob']*100:.1f}%)", fontsize=12)
figA.suptitle("Variant A — predictions on test images", fontsize=14)
figA.tight_layout(rect=[0, 0, 1, 0.96])
outA = ARTIFACTS / "test_predictions_variant_a_3x1.png"
figA.savefig(outA, dpi=150); plt.close(figA)

print(f"\nSaved:\n  {outB}\n  {outA}")
