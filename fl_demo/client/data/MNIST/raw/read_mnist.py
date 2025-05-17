import struct
import numpy as np
import matplotlib.pyplot as plt

def read_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def read_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Đường dẫn tới file MNIST
images = read_images('train-images-idx3-ubyte')
labels = read_labels('train-labels-idx1-ubyte')

# Hiển thị 5 ảnh đầu tiên
for i in range(5):
    plt.imshow(images[i], cmap='gray')
    plt.title(f'Label: {labels[i]}')
    plt.show()