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


x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(f))
print(type(y))
print(y.data)
