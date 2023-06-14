import numpy as np

T, H = 5, 4
hs = np.random.randn(T, H) # 각 단어에 해당하는 벡터들의 집합
a = np.array([0.8, 0.1, 0.03, 0.05, 0.02]) # 각 단어의 중요도를 나타내는 가중치

# repeat() 대신 넘파이의 브로드캐스트 사용 가능
ar = a.reshape(5, 1).repeat(4, axis=1)
print(ar.shape)

t = hs * ar # 단어별 가중치 적용
print(t.shape)

c = np.sum(t, axis=0) # 가중합
print(c.shape)