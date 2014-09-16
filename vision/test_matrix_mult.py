import numpy as np

A = np.random.rand(2, 3)
#B = np.eye(2)
B = np.array([1,0.5,1])

C = A * B

print A
print B
print C
