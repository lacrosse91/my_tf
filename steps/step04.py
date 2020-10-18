import numpy as np

class Variable:
    def __init__(self, data):
        """
        初期化
        """
        self.data = data


class Function:
    def __call__(self, input):
        """
        docstring
        """
        x = input.data # データを取り出す
        y = self.forward(x) #　計算を別のメソッドで行う
        output = Variable(y) # Variableに変換
        return output

    def forward(self, x):
        """
        docstring
        """
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)
