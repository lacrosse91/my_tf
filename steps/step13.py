import numpy as np
import unittest

class Variable:
    def __init__(self, data):
        """
        ndarrayのみdataとして許可
        """
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported' .format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx
                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    def __call__(self, *inputs):
        """
        docstring
        """
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) #　計算を別のメソッドで行う
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] # Variableに変換
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        """
        docstring
        """
        raise NotImplementedError()

    def backward(self, gys):
        """
        docstring
        """
        raise NotImplementedError()

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)

x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)

## テスト

class AddTest(unittest.TestCase):

    def test_forward(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(x0, x1)
        expected = np.array(5)
        self.assertEqual(y.data, expected)

class SquareTest(unittest.TestCase):

    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
