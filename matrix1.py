import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ==============================
# CONFIG: define the class order for each model
# ==============================
CLASS_ORDER_RBC = [
    "BASOcropped",
    "double rbc",
    "EOS IMAGEScropped",
    "LYMcropped",
    "MONOcropped",
    "NEUcropped",
    "single rbc",
    "triple rbc"
]

CLASS_ORDER_WBC = [
    "Baso",
    "Eos",
    "Lympho",
    "Mono",
    "Neutro"
]

# ==============================
# FUNCTION: Evaluate Model
# ==============================
def evaluate_model(model_path, test_dir, class_order, input_shape=(299, 299), batch_size=32):
    print(f"\n🔹 Evaluating model: {model_path}")
    print(f"   Using class order: {class_order}")

    model = load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=class_order
    )
    print("Test Generator Mapping:", test_generator.class_indices)

    # Predictions
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_generator.classes
    labels = list(test_generator.class_indices.keys())

    # Adjust mismatch in output neurons vs folder count
    n_model_classes = Y_pred.shape[1]
    n_folders = len(labels)
    if n_model_classes != n_folders:
        print(f"⚠️ Class mismatch: model has {n_model_classes}, folders have {n_folders}. Trimming to match.")
        n_common = min(n_model_classes, n_folders)
        labels = labels[:n_common]
        Y_pred = Y_pred[:, :n_common]
        y_pred = np.clip(y_pred, 0, n_common - 1)
        y_true = np.clip(y_true, 0, n_common - 1)

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)

    return cm, labels, report


# ==============================
# FUNCTION: Compare Two Models
# ==============================
def compare_models(model1_path, test1_dir, class_order1,
                   model2_path, test2_dir, class_order2):
    cm1, labels1, report1 = evaluate_model(model1_path, test1_dir, class_order1)
    cm2, labels2, report2 = evaluate_model(model2_path, test2_dir, class_order2)

    # Display confusion matrices side-by-side
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels1, yticklabels=labels1)
    plt.title("v3_wbc_only2.h5 (5-Class WBC Model)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.subplot(1, 2, 2)
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels2, yticklabels=labels2)
    plt.title("v3_wbc_gray.h5 (5-Class WBC Model)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()

    # Accuracy comparison
    print("\n=== 📊 Accuracy Comparison ===")
    print(f"v3_wbc_only2.h5 accuracy: {report1['accuracy']:.2f}")
    print(f"v3_wbc_gray.h5 accuracy: {report2['accuracy']:.2f}")

    # Detailed class-wise metrics for new model
    print("\n=== 🧠 v3_wbc_only1.h5 Class-wise Report ===")
    for cls, metrics in report2.items():
        if cls in class_order2:
            print(f"{cls:15s} → Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1-score']:.2f}")


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    model_rbc = "E:/Project25/J_Blosmia1/preprocess/v3_wbc_only2.h5"
    model_wbc = "E:/Project25/J_Blosmia1/preprocess/v3_wbc_gray.h5"

    test_rbc_dir = "E:/Project25/J_Blosmia1/static/wbc/testset/rbc_wbc/"
    test_wbc_dir = "E:/Project25/J_Blosmia1/static/wbc/testset/wbc_only/"

    compare_models(model_rbc, test_wbc_dir, CLASS_ORDER_WBC,
                   model_wbc, test_wbc_dir, CLASS_ORDER_WBC)
