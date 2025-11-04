# kfold_old_model.py
import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import inception_v3
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------- CONFIG ----------------
DATASET_DIR = r"E:/Project25/J_Blosmia1/static/training/rbc_wbc"
OLD_MODEL_H5 = r"E:/Project25/J_Blosmia1/preprocess/v3_rbc.h5"
OLD_MODEL_SAVE_PREFIX = r"E:/Project25/J_Blosmia1/preprocess/v3_rbc"
IMG_SIZE = (299, 299) #This sets the size of the input images.
BATCH_SIZE = 16 #This defines how many images are processed at once during training.
 #Instead of training the model on one image at a time, it processes 16 images per step.
EPOCHS = 8 #One epoch = one full pass through the entire training dataset.
K = 5 #This is usually for K-Fold Cross Validation.The dataset is split into 5 parts (folds).The model trains on 4 parts.Tests (validates) on the 5th.
SEED = 42 #Ensures same random results

# ---------------- reproducibility ----------------
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------- load old model ----------------
def load_old_model(model_path, num_classes):
    if not os.path.exists(model_path):
        print("Old model path not found:", model_path)
        return None
    model = load_model(model_path)
    if model.output_shape[-1] != num_classes:
        x = model.layers[-2].output
        new_out = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(model.input, new_out)
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
        img = inception_v3.preprocess_input(img)
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024, seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------- main K-Fold ----------------
def run_kfold_old_model():
    paths, labels, class_names = collect_paths_and_labels(DATASET_DIR)
    num_classes = len(class_names)
    print("Found classes:", class_names, "num_samples:", len(paths))

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    fold = 0
    metrics_old = []

    for train_idx, test_idx in skf.split(paths, labels):
        fold += 1
        print(f"\n=== Fold {fold}/{K} ===")

        train_ds = make_dataset_from_indices(paths, labels, train_idx)
        val_ds = make_dataset_from_indices(paths, labels, test_idx, shuffle=False)

        if os.path.exists(OLD_MODEL_H5):
            model = load_old_model(OLD_MODEL_H5, num_classes)
            model.fit(train_ds, epochs=max(1, EPOCHS//2), validation_data=val_ds, verbose=1)

            y_true = []
            y_pred = []
            for x_batch, y_batch in val_ds:
                preds = model.predict(x_batch)
                y_true.extend(y_batch.numpy().tolist())
                y_pred.extend(np.argmax(preds, axis=1).tolist())

            acc = accuracy_score(y_true, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            print(f"Old model fold metrics: acc {acc:.4f}, precision {p:.4f}, recall {r:.4f}, f1 {f1:.4f}")
            metrics_old.append((acc, p, r, f1))

            model.save(f"{OLD_MODEL_SAVE_PREFIX}_fold{fold}.h5")
        else:
            print("Old model .h5 not found; skipping fold.")
            metrics_old.append((np.nan, np.nan, np.nan, np.nan))

    # summarize
    arr = np.array(metrics_old, dtype=np.float32)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    print(f"\nOLD MODEL SUMMARY (mean ± std) -> Acc: {mean[0]:.4f} ± {std[0]:.4f}, "
          f"Precision: {mean[1]:.4f} ± {std[1]:.4f}, Recall: {mean[2]:.4f} ± {std[2]:.4f}, "
          f"F1: {mean[3]:.4f} ± {std[3]:.4f}")
    return metrics_old, class_names

if __name__ == "__main__":
    run_kfold_old_model()
