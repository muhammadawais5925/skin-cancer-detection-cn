"""
Skin cancer CNN training.
Dataset structure (class-wise folders):
  skin_cancer_dataset/
      akiec/ bcc/ bkl/ df/ mel/ nv/ vasc/
Chalane ka tareeqa:  python train.py
Output: skin_cancer_cnn.keras  aur  class_names.json
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = Path("skin_cancer_dataset")
IMG_SIZE = (128, 128)   # app.py me bhi yehi size hai
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42
MODEL_PATH = "skin_cancer_cnn.keras"
CLASS_NAMES_PATH = "class_names.json"

if not DATA_DIR.exists():
    raise SystemExit(f"'{DATA_DIR}' folder nahi mila. Pehle prepare_dataset.py chalao ya dataset yahan rakho.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical",
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical",
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Class imbalance (HAM10000 me 'nv' bohat zyada hai) ke liye weights
counts = np.array([len(list((DATA_DIR / c).glob("*"))) for c in class_names], dtype=float)
weights = counts.sum() / (num_classes * counts)
class_weight = {i: float(w) for i, w in enumerate(weights)}
print("Class weights:", class_weight)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Rescaling model ke andar hai, isliye app me raw pixels (0-255) hi bhejne hain
model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)

model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    class_weight=class_weight, callbacks=[early_stop],
)

model.save(MODEL_PATH)
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

print(f"Saved: {MODEL_PATH} and {CLASS_NAMES_PATH}")
