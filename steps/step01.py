import numpy as np

class Variable:
    def __init__(self, data):
        """
        初期化
        """
        self.data = data


data = np.array(1.0)

x = Variable(data)
print(x.data)
