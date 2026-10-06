import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

TEST_DIR = "dataset_split/test"
MODEL_PATH = "models/image_classifier.keras"

os.makedirs("results", exist_ok=True)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("Classes:", class_names)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("\nClassification Report:")
print(report)

with open("results/classification_report.txt", "w") as file:
    file.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(7, 7))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=150
)

plt.close()

print("\nEvaluation completed successfully!")
print("Results saved in the results folder.")