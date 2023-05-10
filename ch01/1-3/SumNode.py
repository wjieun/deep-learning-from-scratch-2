import numpy as np
D, N = 8, 7
dy = np.random.randn(N, D) # 입력
dx = np.sum(dy, axis=0, keepdims=True) # Sum 노드의 순전파
x = np.random.randn(1, D) # 무작위 기울기
y = np.repeat(x, N, axis=0) # Sum 노드의 역전파