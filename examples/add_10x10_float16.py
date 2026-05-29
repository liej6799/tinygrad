from tinygrad import Tensor, dtypes

a = Tensor.full((10, 10), 1.5, dtype=dtypes.float16)
b = Tensor.full((10, 10), 2.25, dtype=dtypes.float16)
c = (a + b).realize()

print(c.dtype, c.shape)
print(c.numpy())
