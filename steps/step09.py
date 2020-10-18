import numpy as np

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
    def __call__(self, input):
        """
        docstring
        """
        x = input.data # データを取り出す
        y = self.forward(x) #　計算を別のメソッドで行う
        output = Variable(as_array(y)) # Variableに変換
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        """
        docstring
        """
        raise NotImplementedError()

    def backward(self, gy):
        """
        docstring
        """
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    return Square()(x)  # 1行でまとめて書く

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)


x = Variable(np.array(1.0))  # OK
x = Variable(None)  # OK
x = Variable(1.0)  # NG
