import numpy as np

A = np.array([
    [1, -1, 2],
    [3, 2, 2]
])
A_T = A.T

B = np.array([
    [0, 1],
    [-1, 1],
    [5, 2]
])
B_T = B.T

C = np.array([
    [2, -1],
    [-3, 3]
])
C_inverse = np.linalg.pinv(C)

D = np.array([
    [-5, 1],
    [2, 0]
])
D_T = D.T

# 행렬 연산을 result 변수에 저장하세요
result = B_T @ (2*A_T) @ (3*C_inverse + D_T)

result