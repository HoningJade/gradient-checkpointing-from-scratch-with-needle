from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND


class LogSoftmax(TensorOp):
    def compute(self, Z):
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes, )
        self.axes = axes

    def compute(self, Z):
        # BEGIN YOUR SOLUTION
        if self.axes == (-1, ):
            self.axes = (len(Z.shape)-1, )
        z = Z.max(axis=self.axes, keepdims=True)
        z_broadcast = array_api.broadcast_to(z, Z.shape)
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(
            Z - z_broadcast), axis=self.axes, keepdims=True)) + z

        new_shape = []
        if self.axes:
            l = len(Z.shape)
            for i, n in enumerate(Z.shape):
                if (i not in self.axes) and ((i - l) not in self.axes):
                    new_shape.append(n)
            log_sum_exp = log_sum_exp.reshape(new_shape)  # .astype(Z.dtype)
        else:
            self.reduced_shape = (1,) * len(Z.shape)
        return log_sum_exp
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        input, = node.inputs
        data = input.realize_cached_data()
        z = data.max(axis=self.axes, keepdims=True)
        z = array_api.broadcast_to(z, input.shape)
        e = array_api.exp(data - z)
        e_sum = array_api.sum(e, axis=self.axes, keepdims=True)
        e_sum = array_api.broadcast_to(e_sum, input.shape)
        prob = e / e_sum
        new_shape = list(input.shape)
        # (a, b) -> (1, a, 1, b)
        if self.axes:
            for i in self.axes:
                new_shape[i] = 1
            grad = reshape(out_grad, new_shape)
        else:
            grad = out_grad
        return broadcast_to(grad, input.shape) * Tensor(prob, dtype=grad.dtype, device=grad.device)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
