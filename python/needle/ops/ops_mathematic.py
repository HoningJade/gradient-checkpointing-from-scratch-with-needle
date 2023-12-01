"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from ..init import *

import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # BEGIN YOUR SOLUTION
        return a**self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # BEGIN YOUR SOLUTION
        node_value = node.inputs[0]
        return out_grad * (self.scalar * (node_value ** (self.scalar - 1)))
        # END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return a / b
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * (-1.0 * lhs / (rhs**2))
        # END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a / self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        # END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: Tensor):
        # BEGIN YOUR SOLUTION
        shape = list(range(len(a.shape)))
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        shape[x], shape[y] = shape[y], shape[x]

        return a.permute(tuple(shape))
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        axes = self.axes or (-1, -2)
        return transpose(out_grad, axes)
        # END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        node_value = node.inputs[0]
        return reshape(out_grad, node_value.shape)
        # END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        node_value = node.inputs[0]
        n1 = len(node_value.shape)
        n2 = len(self.shape)

        axes = list(range(n2 - n1))
        # broadcasted
        axes += list(reversed([i for i in range(n1) if node_value.shape[i] == 1]))
        return reshape(summation(out_grad, axes=tuple(axes)), node_value.shape)
        # END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        n = len(a.shape)
        axes = []
        if not isinstance(self.axes, tuple):
            ori_axes = (self.axes,)
        else:
            ori_axes = self.axes
        for axis in ori_axes:
            if isinstance(axis, int):
                if axis < 0:
                    axes.append(axis + n)
                else:
                    axes.append(axis)
            else:
                axes.append(axis)
        axes = sorted(axes, reverse=True)
        for axis in axes:
            a = array_api.sum(a, axis)

        return a
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        (input,) = node.inputs
        if self.axes == None:
            axes = input.shape
            grad_shape = []
        else:
            axes = self.axes
            grad_shape = list(out_grad.shape)

        n = len(input.shape)
        new_axes = []
        for x in axes:
            if x >= 0:
                new_axes.append(x)
            else:
                new_axes.append(x + n)
        new_axes = sorted(new_axes)
        for axis in new_axes:
            grad_shape.insert(axis, 1)

        return broadcast_to(reshape(out_grad, grad_shape), input.shape)
        # END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # BEGIN YOUR SOLUTION

        return a @ b
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs, (-1, -2)))
        rhs_grad = matmul(transpose(lhs, (-1, -2)), out_grad)
        grad_dim = len(out_grad.shape)
        lhs_dim = len(lhs.shape)
        rhs_dim = len(rhs.shape)

        if grad_dim > lhs_dim:
            lhs_grad = summation(lhs_grad, axes=tuple(range(grad_dim - lhs_dim)))
        if grad_dim > rhs_dim:
            rhs_grad = summation(rhs_grad, axes=tuple(range(grad_dim - rhs_dim)))

        return lhs_grad, rhs_grad
        # END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return -a
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return -out_grad
        # END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.log(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        # END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        # END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        node_value = node.inputs[0]
        return out_grad * Tensor(
            node_value.realize_cached_data() > 0,
            device=node.device,
            dtype=node.dtype,
            required_grad=node.requires_grad,
        )
        # END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        (input,) = node.inputs
        tmp = tanh(input)
        one_arr = ones(*out_grad.shape, device=out_grad.device, requires_grad=False)
        return out_grad * (one_arr - tmp * tmp)
        # END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        # BEGIN YOUR SOLUTION
        n = len(args)
        shape = list(args[0].shape)
        dtype = args[0].dtype
        device = args[0].device
        arg_shape = list(args[0].shape)
        shape.insert(self.axis, n)
        new_arr = array_api.empty(shape, dtype=dtype, device=device)
        idxes = []
        m = len(arg_shape)
        for i in range(m):
            idxes.append(slice(0, arg_shape[i]))
        idxes.insert(self.axis, 0)
        arg_shape.insert(self.axis, 1)

        for i in range(n):
            idxes[self.axis] = i
            new_arr[tuple(idxes)] = array_api.reshape(args[i], arg_shape)

        return new_arr
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return split(out_grad, axis=self.axis)
        # END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        # BEGIN YOUR SOLUTION
        n = A.shape[self.axis]
        arg_shape = list(A.shape)
        new_arr = []
        idxes = []
        m = len(arg_shape)
        for i in range(m):
            idxes.append(slice(0, arg_shape[i]))
        new_shape = list(A.shape)
        del new_shape[self.axis]

        for i in range(n):
            idxes[self.axis] = i
            data = array_api.array(A[tuple(idxes)], dtype=A.dtype, device=A.device)
            data = array_api.reshape(data, new_shape)
            new_arr.append(data)

        return new_arr
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return stack(out_grad, axis=self.axis)
        # END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        # END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        old_shape = list(a.shape)
        n = len(old_shape)
        new_shape = []
        index = []
        for i in range(n):
            if i not in self.axes:
                new_shape.append(old_shape[i])
                index.append(slice(new_shape[-1]))
            else:
                new_shape.append(old_shape[i] * (1 + self.dilation))
                index.append(slice(0, new_shape[-1], 1 + self.dilation))

        res = array_api.full(new_shape, 0, dtype=a.dtype, device=a.device)
        res[tuple(index)] = a

        return res
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        # END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        old_shape = list(a.shape)
        n = len(old_shape)
        new_shape = []
        index = []
        for i in range(n):
            if i not in self.axes:
                new_shape.append(old_shape[i])
                index.append(slice(new_shape[-1]))
            else:
                new_shape.append(old_shape[i] // (1 + self.dilation))
                index.append(slice(0, old_shape[i], 1 + self.dilation))

        res = array_api.full(new_shape, 0, dtype=a.dtype, device=a.device)
        res = a[tuple(index)]

        return res
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        # END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        # BEGIN YOUR SOLUTION
        # Do not pad batch and channel dimensions.
        axes = (
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
            (0, 0),
        )
        A = A.pad(axes)
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape  # square kernel by convention
        Ns, Hs, Ws, Cs = A.strides

        _H, _W = (H - K) // self.stride + 1, (W - K) // self.stride + 1
        inner_dim = K * K * C_in
        _A = A.as_strided(
            shape=(N, _H, _W, K, K, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
        ).compact()
        _A = _A.reshape((-1, inner_dim))
        B = B.compact().reshape((-1, C_out))
        out = _A @ B
        return out.reshape((N, _H, _W, C_out))
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        out_grad_dilate = dilate(out_grad, (1, 2), self.stride - 1)
        # A: bhwc1, B: kkc1c2
        A, B = node.inputs
        A = A.realize_cached_data()
        B = B.realize_cached_data()
        b = A.shape[0]
        h = A.shape[1]
        k = B.shape[0]
        # bhwc1 -> c1hwb
        A = A.permute((3, 1, 2, 0))
        # kkc1c2 -> kkc1c2
        B = array_api.flip(B, (0, 1))
        # kkc1c2 -> kkc2c1
        B = B.permute((0, 1, 3, 2))
        tmp = ((h + 2 * self.padding - k) // self.stride + 1) * self.stride
        # pad
        p_B = (h + k - tmp - 1) // 2
        p_A = (k + tmp - h - 1) // 2
        # bhwc2, kkc2c1 -> bhwc1
        grad_A = conv(
            out_grad_dilate,
            Tensor(B, dtype=out_grad.dtype, device=out_grad.device),
            stride=1,
            padding=p_B,
        )
        # bhwc2 -> whbc2 -> hwbc2: out_grad_dilate.transpose((0, 2)).transpose((0, 1))
        # c1hwb, hwbc2 -> c1hwc2
        grad_B = conv(
            Tensor(A, dtype=out_grad.dtype, device=out_grad.device),
            out_grad_dilate.transpose((0, 2)).transpose((0, 1)),
            stride=1,
            padding=p_A,
        )
        # c1hwc2 -> whc1c2 -> hwc1c2
        grad_B = grad_B.transpose((0, 2)).transpose((0, 1))

        return grad_A, grad_B
        # END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
