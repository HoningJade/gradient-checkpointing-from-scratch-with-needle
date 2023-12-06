"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

# utils for gradient checkpointing
def annotate(out_node, in_node):
    if out_node is in_node:
        return 
    
    for node in out_node.inputs:
        node.drop = node.op is not None
        annotate(node, in_node)
            
class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True
        self.gc = False

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True
    
    def enable_gc(self):
        self.gc = True
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            in_features, out_features), device=device, dtype=dtype)
        self.bias = Parameter(ops.reshape(init.kaiming_uniform(
            out_features, 1), (1, out_features)), device=device, dtype=dtype) if bias else None
        # END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        Y = ops.matmul(X, self.weight)
        if self.bias:
            bias = ops.broadcast_to(self.bias, Y.shape)
            Y += bias
        return Y
        # END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        # BEGIN YOUR SOLUTION
        shape = 1
        for dim in X.shape[1:]:
            shape *= dim
        return X.reshape((X.shape[0], shape))
        # END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        output = ops.relu(x)
        if self.gc:
            annotate(output, x)
        return output
        # END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        # END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # BEGIN YOUR SOLUTION
        n = logits.shape[0]
        m = logits.shape[-1]
        y_one_hot = init.one_hot(
            m, y, device=logits.device, dtype=logits.dtype, requires_grad=False)
        z_y = ops.summation(logits * y_one_hot, axes=(-1, ))
        return ops.summation(ops.logsumexp(logits, axes=(-1,)) - z_y) / float(n)
        # END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # BEGIN YOUR SOLUTION
        w = init.ones(dim, device=device, dtype=dtype)
        self.weight = Parameter(w, device=device, dtype=dtype)
        b = init.zeros(dim, device=device, dtype=dtype)
        self.bias = Parameter(b, device=device, dtype=dtype)

        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        num_dim = len(x.shape)
        d = x.shape[-1]
        n = x.shape[-2]
        broadcast_shape = tuple([1] * (len(x.shape) - 1) + [d])
        stats_shape = x.shape[:-2] + (1, d)

        # constant
        c = 1
        for dim in x.shape[:-1]:
            c *= dim
        if self.training:
            x_mean = ops.summation(x, axes=-2) / n
            x_mean = ops.broadcast_to(x_mean.reshape(stats_shape), x.shape)

            # moving average
            running_mean = ops.broadcast_to(
                self.running_mean.reshape(broadcast_shape), x.shape).data
            running_mean = (1 - self.momentum) * \
                running_mean.data + self.momentum * x_mean.data
            self.running_mean = ops.summation(
                running_mean, axes=tuple(range(num_dim - 1))).data / c

            x_var = ops.summation((x - x_mean) ** 2, axes=-2) / n
            x_var = ops.broadcast_to(x_var.reshape(stats_shape), x.shape)
            # moving average
            running_var = ops.broadcast_to(ops.reshape(
                self.running_var, broadcast_shape), x.shape).data
            running_var = (1 - self.momentum) * \
                running_var.data + self.momentum * x_var.data
            self.running_var = ops.summation(
                running_var, axes=tuple(range(num_dim - 1))).data / c
        else:
            x_mean = ops.broadcast_to(
                self.running_mean.reshape(broadcast_shape), x.shape)
            x_var = ops.broadcast_to(
                self.running_var.reshape(broadcast_shape), x.shape)

        # normalize
        x_normalize = (x - x_mean) / ops.power_scalar(x_var + self.eps, 0.5)

        weight = ops.broadcast_to(
            self.weight.reshape(broadcast_shape), x.shape)
        bias = ops.broadcast_to(self.bias.reshape(broadcast_shape), x.shape)
        return weight * x_normalize + bias


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose(
            (2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32", gc=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # BEGIN YOUR SOLUTION
        w = init.ones(dim)
        self.weight = Parameter(w, device=device, dtype=dtype)
        b = init.zeros(dim)
        self.bias = Parameter(b, device=device, dtype=dtype)
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        d = x.shape[-1]
        broadcast_shape = tuple([1] * (len(x.shape) - 1) + [d])
        stats_shape = tuple(list(x.shape)[:-1] + [1])

        x_mean = ops.summation(x, axes=-1) / x.shape[-1]
        x_mean = ops.broadcast_to(x_mean.reshape(stats_shape), x.shape)

        x_var = ops.summation((x - x_mean) ** 2, axes=-1) / d
        x_var = ops.broadcast_to(x_var.reshape(stats_shape), x.shape)
        x_normalized = (x - x_mean) / ops.power_scalar(x_var + self.eps, 0.5)

        weight = ops.broadcast_to(
            self.weight.reshape(broadcast_shape), x.shape)
        bias = ops.broadcast_to(self.bias.reshape(broadcast_shape), x.shape)
        output = weight * x_normalized + bias
        
        if self.gc:
            annotate(output, x)
        
        return output
        # END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        if self.training:
            prob = init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype)
            return ops.multiply(x, prob) / (1 - self.p)
        else:
            return x
        # END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return self.fn(x) + x
        # END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return ops.tanh(x)
        # END YOUR SOLUTION
