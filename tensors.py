import numpy as np
# 0 D tensor/scalar
a = np.array(4)
print(a.shape)
print(a.ndim) # shows no of dimesnions

# 1D tensor/vector
b = np.array([1,2,3,4])
print(b.shape)
print(b.ndim)

# 2D tensor/matrix
c = np.array([[1,2,3],[4,5,6]])
print(c)
print(c.shape) # (2*3)
print(c.ndim)

# 3D tensor
d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]])
print(d)
print(d.shape)
print(d.ndim)
