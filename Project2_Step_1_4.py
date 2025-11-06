import os, json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
IMAGE_SIZE = (500, 500)
INPUT_SHAPE = (500, 500, 3)
BATCH_SIZE  = 32
EPOCHS      = int(os.getenv("PROJECT2_EPOCHS", "30"))
SEED        = 42

BASE      = Path(__file__).resolve().parent
DATA_DIR  = BASE / "Project 2 Data" / "Data"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
ARTIFACTS = BASE / "artifacts"
MODELS    = BASE / "models"
for d in (ARTIFACTS, MODELS): d.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED); tf.random.set_seed(SEED)

# allow GPU memory to grow (avoids big upfront grab)
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass


print("Config:", dict(IMAGE_SIZE=IMAGE_SIZE, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS))

# --- data ---
print("\nStep 1 - Data")
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
rows = []
for name, root in (("train", TRAIN_DIR), ("valid", VALID_DIR)):
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        cnt = sum(1 for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in img_exts)
        rows.append({"split": name, "class": cdir.name, "count": cnt})
df = pd.DataFrame(rows)
if df.empty: raise RuntimeError("No images found.")
print(df.pivot_table(index="class", columns="split", values="count", fill_value=0))

train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    shear_range=0.10, zoom_range=0.10,
    rotation_range=10, width_shift_range=0.05, height_shift_range=0.05
)
valid_datagen = ImageDataGenerator(rescale=1/255.0)

train_gen = train_datagen.flow_from_directory(
    str(TRAIN_DIR), target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True, seed=SEED
)
valid_gen = valid_datagen.flow_from_directory(
    str(VALID_DIR), target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

class_indices = train_gen.class_indices
labels = [lbl for lbl, idx in sorted(class_indices.items(), key=lambda x: x[1])]
print("Classes:", class_indices)

def _plot_confusion(cm, labels, title, out_path, cmap):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

# --- variant A ---
print("\nStep 2 - Variant A")
tf.keras.backend.clear_session()
model_a = models.Sequential(name="variant_a")
model_a.add(layers.Input(shape=INPUT_SHAPE))
model_a.add(layers.Conv2D(32, (3,3), activation="relu", padding="same")); model_a.add(layers.MaxPooling2D((2,2)))
model_a.add(layers.Conv2D(64, (3,3), activation="relu", padding="same")); model_a.add(layers.MaxPooling2D((2,2)))
model_a.add(layers.Conv2D(128,(3,3), activation="relu", padding="same")); model_a.add(layers.MaxPooling2D((2,2)))
model_a.add(layers.Conv2D(256,(3,3), activation="relu", padding="same")); model_a.add(layers.MaxPooling2D((2,2)))
model_a.add(layers.Flatten())
model_a.add(layers.Dense(128, activation="relu"))
model_a.add(layers.Dropout(0.5))  # a bit more dropout to curb early overfit
model_a.add(layers.Dense(3, activation="softmax"))
model_a.summary()

model_a.compile(optimizer=optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
cb_a = [
    # save best by val-accuracy 
    callbacks.ModelCheckpoint(str(MODELS/"variant_a_best.keras"), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
    # stop based on val-loss (smoother than accuracy)
    callbacks.EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-3, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]
hist_a = model_a.fit(train_gen, epochs=EPOCHS, validation_data=valid_gen, callbacks=cb_a, verbose=1)

# eval A
print("\nStep 3 - Evaluate A")
valid_gen.reset()
eval_a = model_a.evaluate(valid_gen, verbose=1)
valid_gen.reset()
pred_a = model_a.predict(valid_gen, verbose=1)
y_pred_a = np.argmax(pred_a, axis=1)
cm_a = np.zeros((len(labels), len(labels)), dtype=int)
for t, p in zip(valid_gen.classes, y_pred_a): cm_a[t, p] += 1
row_a = cm_a.sum(axis=1)
pacc_a = np.divide(np.diag(cm_a), row_a, out=np.zeros_like(row_a, dtype=float), where=row_a!=0)
print("A  val_loss/val_acc:", eval_a)
for lbl, sc in zip(labels, pacc_a): print(f"A  per-class {lbl}: {sc:.3f}")

histdf_a = pd.DataFrame(hist_a.history); ep_a = np.arange(1, len(histdf_a)+1)
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].plot(ep_a, histdf_a["accuracy"], label="train"); ax[0].plot(ep_a, histdf_a["val_accuracy"], label="val")
ax[0].set_title("Accuracy (A)"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(ep_a, histdf_a["loss"], label="train"); ax[1].plot(ep_a, histdf_a["val_loss"], label="val")
ax[1].set_title("Loss (A)"); ax[1].set_xlabel("Epoch"); ax[1].legend()
fig.tight_layout(); fig.savefig(ARTIFACTS/"variant_a_training_curves.png", dpi=150); plt.close(fig)

_plot_confusion(cm_a, labels, "Validation Confusion (A)", ARTIFACTS/"variant_a_val_confusion_matrix.png", "Blues")

best_epoch_a = int(np.argmax(histdf_a["val_accuracy"].values) + 1)
res_a = dict(
    variant="variant_a",
    best_epoch=best_epoch_a,
    best_val_accuracy=float(np.max(histdf_a["val_accuracy"].values)),
    best_val_loss=float(np.min(histdf_a["val_loss"].values)),
    final_val_accuracy=float(histdf_a["val_accuracy"].values[-1]),
    final_val_loss=float(histdf_a["val_loss"].values[-1]),
    saved_model=str(MODELS/"variant_a_best.keras"),
)

# --- variant B ---
print("\nStep 2 - Variant B")
tf.keras.backend.clear_session()
model_b = models.Sequential(name="variant_b")
model_b.add(layers.Input(shape=INPUT_SHAPE))
# 3x3 at full res; 5x5 after first pool to cut compute
model_b.add(layers.Conv2D(48, (3,3), padding="same")); model_b.add(layers.LeakyReLU(negative_slope=0.1)); model_b.add(layers.MaxPooling2D((2,2)))
model_b.add(layers.Conv2D(96, (5,5), padding="same"));  model_b.add(layers.LeakyReLU(negative_slope=0.1)); model_b.add(layers.MaxPooling2D((2,2)))
model_b.add(layers.Conv2D(192,(3,3), padding="same"));  model_b.add(layers.LeakyReLU(negative_slope=0.1)); model_b.add(layers.MaxPooling2D((2,2)))
model_b.add(layers.Conv2D(256,(3,3), padding="same"));  model_b.add(layers.LeakyReLU(negative_slope=0.1)); model_b.add(layers.MaxPooling2D((2,2)))
model_b.add(layers.Conv2D(320,(3,3), padding="same"));  model_b.add(layers.LeakyReLU(negative_slope=0.1)); model_b.add(layers.MaxPooling2D((2,2)))
model_b.add(layers.Flatten())
model_b.add(layers.Dense(192)); model_b.add(layers.ELU()); model_b.add(layers.Dropout(0.5))
model_b.add(layers.Dense(3, activation="softmax"))
model_b.summary()

model_b.compile(optimizer=optimizers.Adam(8e-5), loss="categorical_crossentropy", metrics=["accuracy"])
cb_b = [
    callbacks.ModelCheckpoint(str(MODELS/"variant_b_best.keras"), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-3, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]
hist_b = model_b.fit(train_gen, epochs=EPOCHS, validation_data=valid_gen, callbacks=cb_b, verbose=1)

# eval B
print("\nStep 3 - Evaluate B")
valid_gen.reset()
eval_b = model_b.evaluate(valid_gen, verbose=1)
valid_gen.reset()
pred_b = model_b.predict(valid_gen, verbose=1)
y_pred_b = np.argmax(pred_b, axis=1)
cm_b = np.zeros((len(labels), len(labels)), dtype=int)
for t, p in zip(valid_gen.classes, y_pred_b): cm_b[t, p] += 1
row_b = cm_b.sum(axis=1)
pacc_b = np.divide(np.diag(cm_b), row_b, out=np.zeros_like(row_b, dtype=float), where=row_b!=0)
print("B  val_loss/val_acc:", eval_b)
for lbl, sc in zip(labels, pacc_b): print(f"B  per-class {lbl}: {sc:.3f}")

histdf_b = pd.DataFrame(hist_b.history); ep_b = np.arange(1, len(histdf_b)+1)
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].plot(ep_b, histdf_b["accuracy"], label="train"); ax[0].plot(ep_b, histdf_b["val_accuracy"], label="val")
ax[0].set_title("Accuracy (B)"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(ep_b, histdf_b["loss"], label="train"); ax[1].plot(ep_b, histdf_b["val_loss"], label="val")
ax[1].set_title("Loss (B)"); ax[1].set_xlabel("Epoch"); ax[1].legend()
fig.tight_layout(); fig.savefig(ARTIFACTS/"variant_b_training_curves.png", dpi=150); plt.close(fig)

_plot_confusion(cm_b, labels, "Validation Confusion (B)", ARTIFACTS/"variant_b_val_confusion_matrix.png", "Purples")

best_epoch_b = int(np.argmax(histdf_b["val_accuracy"].values) + 1)
res_b = dict(
    variant="variant_b",
    best_epoch=best_epoch_b,
    best_val_accuracy=float(np.max(histdf_b["val_accuracy"].values)),
    best_val_loss=float(np.min(histdf_b["val_loss"].values)),
    final_val_accuracy=float(histdf_b["val_accuracy"].values[-1]),
    final_val_loss=float(histdf_b["val_loss"].values[-1]),
    saved_model=str(MODELS/"variant_b_best.keras"),
)

# --- summary ---
print("\nStep 4 - Summary")
comparison_df = pd.DataFrame([res_a, res_b])
print(comparison_df.to_string(index=False))
best = comparison_df.loc[comparison_df["best_val_accuracy"].idxmax()]

label_to_index = {lbl:int(idx) for lbl, idx in class_indices.items()}
index_to_label = {str(idx): lbl for lbl, idx in class_indices.items()}
summary = {
    "best_variant": best["variant"],
    "best_model_path": best["saved_model"],
    "best_val_accuracy": float(best["best_val_accuracy"]),
    "best_val_loss": float(best["best_val_loss"]),
    "class_indices": {"label_to_index": label_to_index, "index_to_label": index_to_label},
    "comparison_table": {"variant_a": res_a, "variant_b": res_b},
    "timestamp_utc": datetime.utcnow().isoformat(),
}
with open(ARTIFACTS/"training_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: {ARTIFACTS/'training_summary.json'}")
print(f"Best: {summary['best_variant']}  ->  {summary['best_model_path']}")
