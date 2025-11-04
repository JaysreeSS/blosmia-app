# kfold_new_model.py
import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------- CONFIG ----------------
DATASET_DIR = r"E:/Project25/J_Blosmia1/static/training/rbc_wbc"
NEW_MODEL_SAVE_PREFIX = r"E:/Project25/J_Blosmia1/preprocess/v3_wbc_only2.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8
K = 5
SEED = 42

# ---------------- reproducibility ----------------
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------- build new model ----------------
def build_new_model(input_shape=(*IMG_SIZE, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ---------------- collect paths and labels ----------------
def collect_paths_and_labels(dataset_dir):
    filepaths = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    for c in class_names:
        cdir = os.path.join(dataset_dir, c)
        if not os.path.isdir(cdir):
            continue
        for fname in os.listdir(cdir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
                fpath = os.path.join(cdir, fname)
                try:
                    img = Image.open(fpath)
                    img.verify()
                    filepaths.append(fpath)
                    labels.append(class_to_idx[c])
                except:
                    print("Skipping corrupted image:", fpath)
    return np.array(filepaths), np.array(labels), class_names

# ---------------- dataset builder ----------------
def make_dataset_from_indices(paths, labels, idxs, batch_size=BATCH_SIZE,
                              shuffle=True, img_size=IMG_SIZE):
    sel_paths = paths[idxs]
    sel_labels = labels[idxs]
    ds = tf.data.Dataset.from_tensor_slices((sel_paths, sel_labels))

    def _load(x_path, y):
        img_bytes = tf.io.read_file(x_path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, img_size)
        img = resnet50.preprocess_input(img)
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024, seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------- main K-Fold ----------------
def run_kfold_new_model():
    paths, labels, class_names = collect_paths_and_labels(DATASET_DIR)
    num_classes = len(class_names)
    print("Found classes:", class_names, "num_samples:", len(paths))

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    fold = 0
    metrics_new = []

    for train_idx, test_idx in skf.split(paths, labels):
        fold += 1
        print(f"\n=== Fold {fold}/{K} ===")

        train_ds = make_dataset_from_indices(paths, labels, train_idx)
        val_ds = make_dataset_from_indices(paths, labels, test_idx, shuffle=False)

        model = build_new_model(num_classes=num_classes)
        model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1)

        y_true = []
        y_pred = []
        for x_batch, y_batch in val_ds:
            preds = model.predict(x_batch)
            y_true.extend(y_batch.numpy().tolist())
            y_pred.extend(np.argmax(preds, axis=1).tolist())

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        print(f"New model fold metrics: acc {acc:.4f}, precision {p:.4f}, recall {r:.4f}, f1 {f1:.4f}")
        metrics_new.append((acc, p, r, f1))

        model.save(f"{NEW_MODEL_SAVE_PREFIX}_fold{fold}.h5")

    # summarize
    arr = np.array(metrics_new, dtype=np.float32)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    print(f"\nNEW MODEL SUMMARY (mean ± std) -> Acc: {mean[0]:.4f} ± {std[0]:.4f}, "
          f"Precision: {mean[1]:.4f} ± {std[1]:.4f}, Recall: {mean[2]:.4f} ± {std[2]:.4f}, "
          f"F1: {mean[3]:.4f} ± {std[3]:.4f}")
    return metrics_new, class_names

if __name__ == "__main__":
    run_kfold_new_model()
