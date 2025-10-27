import numpy as np
import struct

def readMNIST(label_filepath, image_filepath):
    """
     Reads MNIST data from the IDX file format.

     Args:
         image_filepath (str): Path to the image file (e.g., 'train-images-idx3-ubyte').
         label_filepath (str): Path to the label file (e.g., 'train-labels-idx1-ubyte').

     Returns:
         tuple: A tuple containing (images, labels) as NumPy arrays.

     """
    # Read the labels
    with open(label_filepath, "rb") as f:
        # The '>' denotes big-endian byte order.
        # 'I' is an unsigned integer (4 bytes).
        # We read the magic number and the number of items.
        magic_number, num_items = struct.unpack(">II", f.read(8))
        '''
        struct is a Python library for interpreting bytes as packed binary data. It's for when
        you have a sequence of bytes and you know it represents, for example, an integer followed
        by a float followed by another integer.

        .unpack() takes a format string and a byte string and "unpacks" the bytes into Python variables
        according to the format.
        
        The Format String ">II":
        >: This is the endianness specifier. It stands for big-endian.
        I: This stands for a 4-byte unsigned integer.
        II: This means we expect to find two consecutive 4-byte unsigned integers.
        
        Endianness refers to the order in which a computer stores the bytes of a multi-byte number in memory. 
            Big-Endian: Stores the "big end" (most significant byte) first. This is how humans read numbers.
            Little-Endian: Stores the "little end" (least significant byte) first. 
        Since the MNIST file format was designed to be big-endian, we must explicitly tell our parsing code to 
        read the bytes in that order using the > symbol. If we didn't, struct might use the system's native format 
        (likely little-endian) and get the header numbers completely wrong.
        '''

        if magic_number != 2049:
            raise ValueError(f"Invalid magic number {magic_number} in label file path {label_filepath}")
         # Read the rest of the data into a buffer
        labels = np.frombuffer(f.read(), dtype=np.uint8) #np.frombuffer() is a highly efficient NumPy function. It takes a buffer of
        # raw bytes and creates a NumPy array without making an extra copy of the data.

    # Read the images
    with open(image_filepath, "rb") as f:
        # Read the header information
        magic_number, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        if magic_number != 2051:
            raise ValueError(f"Invalid magic number {magic_number} in image file path {image_filepath}")
        # Read the rest of the image data
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the data into (num_images, rows, cols)
        images = image_data.reshape(num_images, num_rows, num_cols)
    return images, labels