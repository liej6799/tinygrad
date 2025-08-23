from tinygrad import Tensor, dtypes
# rand = Tensor.rand(2, 3) # create a tensor of shape (2, 3) filled with random values from a uniform distribution
# Create two tensors

t1 = Tensor([8.1, 9.2, 6.7], dtype=dtypes.float)
t2 = Tensor([10.4, 7.3, 6.9], dtype=dtypes.float)
# t1 = Tensor.full((1, 1), fill_value=10.0, dtype=dtypes.float32)
# t2 = Tensor.full((1, 1), fill_value=10.0, dtype=dtypes.float32)


t1_float32 = t1.float()
t2_float32 = t2.float()




t6 = t1_float32 + t2_float32 

print(t6.numpy())