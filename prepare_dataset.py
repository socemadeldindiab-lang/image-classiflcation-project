
import os
import shutil
import random

SOURCE_DIR = "dataset"
OUTPUT_DIR = "dataset_split"

CLASSES = ["book", "cup", "mobile"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

for class_name in CLASSES:

    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    images = [
        file for file in os.listdir(source_class_dir)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split_name, split_images in [
        ("train", train_images),
        ("validation", val_images),
        ("test", test_images)
    ]:

        destination_dir = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        os.makedirs(destination_dir, exist_ok=True)

        for image in split_images:
            source = os.path.join(source_class_dir, image)
            destination = os.path.join(destination_dir, image)

            shutil.copy2(source, destination)

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(val_images)}, "
        f"Test={len(test_images)}"
    )

print("\nDataset split completed successfully!")