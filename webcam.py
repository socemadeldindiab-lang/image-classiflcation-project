import cv2
import numpy as np
import tensorflow as tf

# =========================
# Load trained model
# =========================

MODEL_PATH = "models/image_classifier.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# Class names
class_names = ["book", "cup", "mobile"]

# =========================
# Start webcam
# =========================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read webcam.")
        break

    # Prepare image
    image = cv2.resize(frame, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    # Prediction
    predictions = model.predict(image, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100

    predicted_class = class_names[predicted_index]

    # Display result
    text = f"{predicted_class.upper()} - {confidence:.1f}%"

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Image Classification - Press Q to Quit", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()