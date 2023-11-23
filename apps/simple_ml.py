"""hw1/apps/simple_ml.py"""

import time
from apps.models import *
import needle.nn as nn
import needle as ndl
import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")

device = ndl.cpu()


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
        magic_number, num_samples, num_rows, num_cols = \
          read_byte(), read_byte(), read_byte(), read_byte()
        num_samples, num_rows, num_cols = num_samples[0], num_rows[0], num_cols[0]

        struct_fmt = '>' + 'B' * num_samples * num_rows * num_cols
        struct_len = struct.calcsize(struct_fmt)
        struct_unpack = struct.Struct(struct_fmt).unpack_from
        
        data = f.read(struct_len)
        s = struct_unpack(data)
        X = np.array(s, dtype='float32').reshape(num_samples, num_rows * num_cols)
        X /= 255


    with gzip.open(label_filename, "rb") as f:
        magic_number, num_samples = read_byte(), read_byte()
        num_samples = num_samples[0]

        struct_fmt = '>' + 'B' * num_samples
        struct_len = struct.calcsize(struct_fmt)
        struct_unpack = struct.Struct(struct_fmt).unpack_from
        
        data = f.read(struct_len)
        s = struct_unpack(data)
        y = np.array(s, dtype='uint8').reshape(num_samples, )
    return X, y
    # END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    z_y = ndl.summation(ndl.multiply(Z, y_one_hot), axes=(1,))
    log_sum_exp_z = ndl.log(ndl.summation(ndl.exp(Z),  axes=(1,)))
    return ndl.summation(log_sum_exp_z - z_y) / batch_size
    # END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    # BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    y_one_hot = np.zeros((num_examples, num_classes))
    y_one_hot[range(num_examples), y] = 1

    idx = 0
    while idx < num_examples:
        start, end = idx, min(num_examples, idx+batch)
        X_batch = ndl.Tensor(X[start:end])
        y_batch = ndl.Tensor(y_one_hot[start:end])

        pred = ndl.matmul(ndl.relu(ndl.matmul(X_batch, W1)), W2)

        loss = softmax_loss(pred, y_batch)
        loss.backward()

        # IMPT: must convert to numpy in order to get rid of the grad
        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())

        idx += batch
    return W1, W2
    # END YOUR SOLUTION

### CIFAR-10 training ###


def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    correct, total_loss = 0, 0

    if opt is None:
        model.eval()
        for batch in dataloader:
            X, y = batch
            X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            out = model(X)
            loss = loss_fn(out, y)
            correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            total_loss += loss.data.numpy() * y.shape[0]
    else:
        model.train()
        for batch in dataloader:
            opt.reset_grad()
            X, y = batch
            X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            total_loss += loss.data.numpy() * y.shape[0]

    sample_nums = len(dataloader.dataset)
    return correct / sample_nums, total_loss / sample_nums
    # END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                  lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(
            dataloader, model, loss_fn=loss_fn, opt=opt)
        print(f"Epoch: {epoch}, Acc: {avg_acc}, Loss: {avg_loss}")
    # END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_cifar10(
        dataloader, model, loss_fn=loss_fn)
    print(f"Evaluation Acc: {avg_acc}, Evaluation Loss: {avg_loss}")
    # END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
                      clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    f = loss_fn()
    avg_loss = []
    avg_acc = 0
    cnt = 0
    n = data.shape[0]
    i = 0
    while i < n:
        if opt:
            opt.reset_grad()
        # (l, b), (l * b, )
        x, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        b = y.shape[0]
        y_, h = model(x)
        loss = f(y_, y)
        if opt:
            loss.backward()
            opt.step()
        cnt += b
        avg_loss.append(loss.numpy().item() * b)
        avg_acc += np.sum(y_.numpy().argmax(axis=1) == y.numpy())
        i += seq_len

    return avg_acc / cnt, np.sum(avg_loss) / cnt
    # END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
              lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
              device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(
            data=data,
            model=model,
            seq_len=seq_len,
            loss_fn=loss_fn,
            opt=opt,
            clip=clip,
            device=device,
            dtype=dtype,
        )

    return avg_acc, avg_loss
    # END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
                 device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(
        data=data,
        model=model,
        seq_len=seq_len,
        loss_fn=loss_fn,
        opt=None,
        clip=None,
        device=device,
        dtype=dtype,
    )

    return avg_acc, avg_loss
    # END YOUR SOLUTION

# CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
