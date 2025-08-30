from tinygrad import Tensor, dtypes
# rand = Tensor.rand(2, 3) # create a tensor of shape (2, 3) filled with random values from a uniform distribution
# Create two tensors

t1 = Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtypes.int32)
t2 = Tensor([[7, 8, 9], [10, 11, 12]], dtype=dtypes.int32)

# t1 = Tensor([2, 1, 4], dtype=dtypes.float16)
# t2 = Tensor([3, 2, 5], dtype=dtypes.float16)




t6 = t1 * t2 

print(t6.numpy())