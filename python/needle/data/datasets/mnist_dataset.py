from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.X, self.y = parse_mnist(image_filename, label_filename)
        # END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        # BEGIN YOUR SOLUTION
        X, y = self.X[index], self.y[index]
        n = len(X.shape)
        if n == 1:
            X = X.reshape(28, 28, -1)
            X = self.apply_transforms(X)
            X = X.reshape(28, 28, -1)
        else:
            m = X.shape[0]
            X = X.reshape(m, 28, 28, -1)
            for i in range(m):
                X[i] = self.apply_transforms(X[i])
        return X, y
        # END YOUR SOLUTION

    def __len__(self) -> int:
        # BEGIN YOUR SOLUTION
        return self.X.shape[0]
        # END YOUR SOLUTION


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # BEGIN YOUR SOLUTION
    def read_byte():
        struct_fmt = '>l'
        struct_len = struct.calcsize(struct_fmt)
        struct_unpack = struct.Struct(struct_fmt).unpack_from
        data = f.read(struct_len)
        s = struct_unpack(data)
        return s

    with gzip.open(image_filesname, "rb") as f:
        _, num_samples, num_rows, num_cols = \
            read_byte(), read_byte(), read_byte(), read_byte()
        num_samples, num_rows, num_cols = num_samples[0], num_rows[0], num_cols[0]

        struct_fmt = '>' + 'B' * num_samples * num_rows * num_cols
        struct_len = struct.calcsize(struct_fmt)
        struct_unpack = struct.Struct(struct_fmt).unpack_from

        data = f.read(struct_len)
        s = struct_unpack(data)
        X = np.array(s, dtype='float32').reshape(
            num_samples, num_rows * num_cols)
        X /= 255

    with gzip.open(label_filename, "rb") as f:
        _, num_samples = read_byte(), read_byte()
        num_samples = num_samples[0]

        struct_fmt = '>' + 'B' * num_samples
        struct_len = struct.calcsize(struct_fmt)
        struct_unpack = struct.Struct(struct_fmt).unpack_from

        data = f.read(struct_len)
        s = struct_unpack(data)
        y = np.array(s, dtype='uint8').reshape(num_samples, )
    return X, y
    # END YOUR SOLUTION
