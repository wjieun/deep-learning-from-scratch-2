import numpy as np

# 활성화 함수로서의 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(10, 2) # 입력
W1 = np.random.randn(2, 4) # 가중치
b1 = np.random.randn(4) # 편향
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

# b1의 형상은 (4,)이지만 (10, 4)로 브로드캐스트됨
h = np.matmul(x, W1) + b1
print(h)
print(h.shape, '\n')

a = sigmoid(h)
print(a)
print(a.shape, '\n')

s = np.matmul(a, W2) + b2
print(s)
print(s.shape, '\n')