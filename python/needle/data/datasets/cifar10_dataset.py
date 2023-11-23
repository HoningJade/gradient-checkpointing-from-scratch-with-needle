import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        # BEGIN YOUR SOLUTION
        super().__init__(transforms)

        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict[b'data'], dict[b'labels']

        if train:
            files = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5"
            ]
        else:
            files = [
                "test_batch"
            ]
        X = []
        y = []
        for file in files:
            data, labels = unpickle(os.path.join(base_folder, file))
            X.append(data)
            y.append(labels)
        X = np.concatenate(X) / 255.0
        y = np.concatenate(y)
        self.n = X.shape[0]
        X = X.reshape(self.n, -1, 32, 32)
        self.X = X
        self.y = y
        # END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        # BEGIN YOUR SOLUTION
        x = self.apply_transforms(self.X[index])
        y = self.y[index]

        return x, y
        # END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        # BEGIN YOUR SOLUTION
        return self.n
        # END YOUR SOLUTION
