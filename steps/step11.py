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
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, inputs):
        """
        docstring
        """
        xs = [x.data for x in inputs]
        ys = self.forward(xs) #　計算を別のメソッドで行う
        outputs = [Variable(as_array(y)) for y in ys] # Variableに変換
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

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
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)

xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)

## テスト

class AddTest(unittest.TestCase):

    def test_forward(self):
        xs = [Variable(np.array(2)), Variable(np.array(3))]
        f = Add()
        ys = f(xs)
        y = ys[0]
        expected = np.array(5)
        self.assertEqual(y.data, expected)
