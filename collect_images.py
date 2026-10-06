import cv2
import os

# Choose the class
CLASS_NAME = input("Enter class name (book / cup / mobile): ").strip().lower()

# Dataset folder
SAVE_DIR = os.path.join("dataset", CLASS_NAME)

# Create folder if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Number of images
NUM_IMAGES = 40

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("\nCamera started.")
print("Press SPACE to capture an image.")
print("Press Q to quit.\n")

count = len(os.listdir(SAVE_DIR))

while count < NUM_IMAGES:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    # Display information
    display = frame.copy()

    cv2.putText(
        display,
        f"Class: {CLASS_NAME}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Images: {count}/{NUM_IMAGES}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Image Collection - Press SPACE", display)

    key = cv2.waitKey(1) & 0xFF

    # Capture image
    if key == 32:
        filename = os.path.join(
            SAVE_DIR,
            f"{CLASS_NAME}_{count + 1:03d}.jpg"
        )

        cv2.imwrite(filename, frame)

        count += 1
        print(f"Saved: {filename}")

    # Quit
    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"\nFinished! {count} images saved for {CLASS_NAME}.")