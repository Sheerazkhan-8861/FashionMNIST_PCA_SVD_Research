import gzip
import struct
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/raw/fashion")


def load_images(file_path):
    """Load images from a compressed IDX file."""
    with gzip.open(file_path, "rb") as file:
        magic, num_images, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.frombuffer(file.read(), dtype=np.uint8)

    if magic != 2051:
        raise ValueError(f"Invalid image file: {file_path}")

    if images.size != num_images * rows * cols:
        raise ValueError(f"Unexpected image data size: {file_path}")

    return images.reshape(num_images, rows, cols)


def load_labels(file_path):
    """Load labels from a compressed IDX file."""
    with gzip.open(file_path, "rb") as file:
        magic, num_labels = struct.unpack(">II", file.read(8))
        labels = np.frombuffer(file.read(), dtype=np.uint8)

    if magic != 2049:
        raise ValueError(f"Invalid label file: {file_path}")

    if labels.size != num_labels:
        raise ValueError(f"Unexpected label data size: {file_path}")

    return labels


def main():
    X_train_images = load_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    y_train = load_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")

    X_test_images = load_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    y_test = load_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")

    print("FASHION-MNIST DATASET INFORMATION")
    print("-" * 40)
    print("Training images shape:", X_train_images.shape)
    print("Training labels shape:", y_train.shape)
    print("Test images shape:", X_test_images.shape)
    print("Test labels shape:", y_test.shape)
    print("Training pixel data type:", X_train_images.dtype)
    print("Test pixel data type:", X_test_images.dtype)
    print("Pixel intensity range:", X_train_images.min(), "to", X_train_images.max())
    print("Training classes:", np.unique(y_train))
    print("Test classes:", np.unique(y_test))
    print("\nDATASET LOADED SUCCESSFULLY")


if __name__ == "__main__":
    main()
