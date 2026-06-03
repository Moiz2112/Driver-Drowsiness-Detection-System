"""
Train a stronger eye-state classifier for open/closed detection.
This version uses the repository train/test split, light augmentation, and
histogram-equalized preprocessing to make closed-eye detection more stable.
"""

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
IMAGE_SIZE = (24, 24)


def preprocess(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, IMAGE_SIZE)
    image = cv2.equalizeHist(image)
    return image


def augment(image: np.ndarray) -> list[np.ndarray]:
    variants = [image]
    variants.append(cv2.flip(image, 1))
    variants.append(cv2.GaussianBlur(image, (3, 3), 0))
    brighter = cv2.convertScaleAbs(image, alpha=1.08, beta=8)
    darker = cv2.convertScaleAbs(image, alpha=0.92, beta=-8)
    variants.extend([brighter, darker])
    return variants


def load_split(split_name: str, *, do_augment: bool) -> tuple[np.ndarray, np.ndarray]:
    split_dir = DATASET_DIR / split_name
    features: list[np.ndarray] = []
    labels: list[str] = []

    print(f"\nLoading {split_name} split from: {split_dir}")

    for class_name in ["open", "close"]:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing dataset folder: {class_dir}")

        image_count = 0
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            processed = preprocess(image)
            variants = augment(processed) if do_augment else [processed]
            for variant in variants:
                features.append((variant.flatten() / 255.0).astype("float32"))
                labels.append(class_name)
            image_count += 1

        print(f"Loaded {image_count} source images from {split_name}/{class_name}")

    return np.array(features), np.array(labels)


def main() -> None:
    print("=" * 60)
    print("Augmented Extra Trees Eye-State Training")
    print("=" * 60)

    np.random.seed(42)

    X_train, y_train = load_split("train", do_augment=True)
    X_test, y_test = load_split("test", do_augment=False)

    print(f"\nTraining samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print(
        f"Label mapping: "
        f"{dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}"
    )

    model = ExtraTreesClassifier(
        n_estimators=700,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=1,
    )

    print("\nTraining Extra Trees classifier...")
    model.fit(X_train, y_train_encoded)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train_encoded, train_predictions)
    test_accuracy = accuracy_score(y_test_encoded, test_predictions)

    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test_encoded,
            test_predictions,
            target_names=label_encoder.classes_,
            digits=4,
        )
    )

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "cnnCat2.joblib"
    encoder_path = MODEL_DIR / "label_encoder.joblib"

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"Model saved to: {model_path}")
    print(f"Label encoder saved to: {encoder_path}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
