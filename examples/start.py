from tinygrad import Tensor, dtypes
# rand = Tensor.rand(2, 3) # create a tensor of shape (2, 3) filled with random values from a uniform distribution
# Create two tensors

t1 = Tensor([[5, 2, 3, 4], [4, 5, 6, 4]])  # 2D tensor (matrix)
t2 = Tensor([[1, 2, 3, 4], [4, 5, 6, 4]])  # 2D tensor (matrix)


t1_float32 = t1
t2_float32 = t2



# t1 = Tensor.full((1, 16777216), fill_value=10, dtype=dtypes.float32)
# t2 = Tensor.full((1, 16777216), fill_value=10, dtype=dtypes.float32)

t6 = t1_float32 + t2_float32 

print(t6.numpy())