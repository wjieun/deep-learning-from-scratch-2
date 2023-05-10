import numpy as np

# 시그모이드 함수를 클래스로 구현
# 주 변환 처리는 forward(x) 메서드가 담당
class Sigmoid:
    def __init__(self):
        # Sigmod 계층에는 학습하는 매개변수가 따로 없음
        self.params = []

    def forward(self, x):
        return 1/ (1 + np.exp(-x))

# 가중치와 편향은 Affine 계층의 매개변수
# 이 두 매개변수는 신경망이 학습될 때 수시로 갱신
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    # 순전파 처리
    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        return out

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 가중치와 편향 초기화
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 모든 가중치를 리스트에 모음
        # 매개변수 갱신과 매개변수 저장을 손쉽게 처리할 수 있음
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(s)