import numpy as np
import struct

def readMNIST(label_filepath, image_filepath):
    with open(label_filepath, "rb") as f:
        magic_number, num_items = struct.unpack(">II", f.read(8))
        if magic_number != 2049:
            raise ValueError(f"Invalid magic number {magic_number} in label file path {label_filepath}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    with open(image_filepath, "rb") as f:
        magic_number, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        if magic_number != 2051:
            raise ValueError(f"Invalid magic number {magic_number} in image file path {image_filepath}")
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, num_rows, num_cols)
    return images, labels