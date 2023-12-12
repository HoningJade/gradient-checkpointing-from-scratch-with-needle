"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, Tanh, ReLU, annotate
import math


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return ops.exp(-ops.log(ops.add_scalar(ops.exp(-x), 1.0)))
        # END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        # BEGIN YOUR SOLUTION
        self.use_bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = math.sqrt(1 / hidden_size)
        w1 = init.rand(input_size, hidden_size, low=-k, high=k)
        self.W_ih = Parameter(w1, device=device, dtype=dtype)
        w2 = init.rand(hidden_size, hidden_size, low=-k, high=k)
        self.W_hh = Parameter(w2, device=device, dtype=dtype)
        if self.use_bias:
            b1 = init.rand(1, hidden_size, low=-k, high=k)
            self.bias_ih = Parameter(b1, device=device, dtype=dtype)
            b2 = init.rand(1, hidden_size, low=-k, high=k)
            self.bias_hh = Parameter(b2, device=device, dtype=dtype)
        if nonlinearity == "tanh":
            self.f = Tanh()
        else:
            self.f = ReLU()
        # END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        # BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h == None:
            h = Tensor(init.zeros(bs, self.hidden_size),
                       device=X.device, dtype=X.dtype)
        tmp = ops.matmul(X, self.W_ih) + ops.matmul(h, self.W_hh)
        if self.use_bias:
            tmp += ops.broadcast_to(ops.reshape(self.bias_ih, (1, self.hidden_size)), tmp.shape) + \
                ops.broadcast_to(ops.reshape(
                    self.bias_hh, (1, self.hidden_size)), tmp.shape)
        h = self.f(tmp)

        return h
        # END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        # BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        rnn_cells = []
        for i in range(num_layers):
            if i == 0:
                d = input_size
            else:
                d = hidden_size
            rnn_cells.append(RNNCell(d, hidden_size, bias,
                             nonlinearity, device, dtype))
        self.rnn_cells = rnn_cells
        # END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        # BEGIN YOUR SOLUTION
        output = []
        n = X.shape[0]
        bs = X.shape[1]
        # (l, b, e)
        if h0 == None:
            h0 = Tensor(init.zeros(self.num_layers, bs,
                        self.hidden_size), device=X.device, dtype=X.dtype)
        # (l, b, e) -> [(b, e), ... , (b, e)]
        h = ops.split(h0, axis=0)
        # (n, b, e) -> [(b, e), ... , (b, e)]
        X_split = ops.split(X, axis=0)
        h_out = []
        for j in range(self.num_layers):
            h_state = h[j]
            X_state = []
            for i in range(n):
                h_state = self.rnn_cells[j](X_split[i], h_state)
                X_state.append(h_state)
            h_out.append(h_state)
            X_split = X_state

        return ops.stack(X_split, 0), ops.stack(h_out, 0)
        # END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        # BEGIN YOUR SOLUTION
        self.use_bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = math.sqrt(1 / hidden_size)
        w1 = init.rand(input_size, 4*hidden_size, low=-k, high=k)
        self.W_ih = Parameter(w1, device=device, dtype=dtype)
        w2 = init.rand(hidden_size, 4*hidden_size, low=-k, high=k)
        self.W_hh = Parameter(w2, device=device, dtype=dtype)
        if self.use_bias:
            b1 = init.rand(1, 4*hidden_size, low=-k, high=k)
            self.bias_ih = Parameter(b1, device=device, dtype=dtype)
            b2 = init.rand(1, 4*hidden_size, low=-k, high=k)
            self.bias_hh = Parameter(b2, device=device, dtype=dtype)
        self.sigma = Sigmoid()
        self.f = Tanh()
        # END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        # BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h == None:
            h0 = Tensor(init.zeros(bs, self.hidden_size),
                        device=X.device, dtype=X.dtype)
            c0 = Tensor(init.zeros(bs, self.hidden_size),
                        device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        # bs, hidden_size * 4
        tmp = ops.matmul(X, self.W_ih) + ops.matmul(h0, self.W_hh)
        if self.use_bias:
            tmp += ops.broadcast_to(ops.reshape(self.bias_ih, (1, 4 * self.hidden_size)), tmp.shape) + \
                ops.broadcast_to(ops.reshape(
                    self.bias_hh, (1, 4 * self.hidden_size)), tmp.shape)
        tmp = ops.reshape(tmp, (bs, 4, self.hidden_size))
        i, f, g, o = ops.split(tmp, 1)
        i = self.sigma(i)
        f = self.sigma(f)
        g = self.f(g)
        o = self.sigma(o)

        c = f * c0 + i * g
        h = o * self.f(c)

        return h, c
        # END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        # BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        lstm_cells = []
        for i in range(num_layers):
            if i == 0:
                d = input_size
            else:
                d = hidden_size
            lstm_cells.append(LSTMCell(d, hidden_size, bias, device, dtype))
        self.lstm_cells = lstm_cells
        # END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        # BEGIN YOUR SOLUTION
        output = []
        n = X.shape[0]
        bs = X.shape[1]
        # (l, b, e)
        if h == None:
            h0 = Tensor(init.zeros(self.num_layers, bs,
                        self.hidden_size), device=X.device, dtype=X.dtype)
            c0 = Tensor(init.zeros(self.num_layers, bs,
                        self.hidden_size), device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        # (l, b, e) -> [(b, e), ... , (b, e)]
        h = ops.split(h0, axis=0)
        c = ops.split(c0, axis=0)
        # (n, b, e) -> [(b, e), ... , (b, e)]
        X_split = ops.split(X, axis=0)

        h_out = []
        c_out = []
        for j in range(self.num_layers):
            h_state = h[j]
            c_state = c[j]
            X_state = []
            for i in range(n):
                h_state, c_state = self.lstm_cells[j](
                    X_split[i], (h_state, c_state))
                X_state.append(h_state)
            h_out.append(h_state)
            c_out.append(c_state)
            X_split = X_state

        return ops.stack(X_split, 0), (ops.stack(h_out, 0), ops.stack(c_out, 0))
        # END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        # BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight = init.randn(num_embeddings, embedding_dim)
        self.weight = Parameter(weight, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        # BEGIN YOUR SOLUTION
        # l, b, m
        x_one_hot = init.one_hot(
            self.num_embeddings, x, device=x.device, dtype=x.dtype)
        n, b, m = x_one_hot.shape
        # l, b, m -> l * b, m
        x_one_hot = ops.reshape(x_one_hot, (n * b, m))
        # l * b, d
        output = ops.matmul(x_one_hot, self.weight)
        # l, b, d
        output = ops.reshape(output, (n, b, self.embedding_dim))
        
        if self.gc:
            annotate(output, (x, ), segment_len=self.segment_len)

        return output
        # END YOUR SOLUTION
