from tinygrad import Tensor, dtypes
# rand = Tensor.rand(2, 3) # create a tensor of shape (2, 3) filled with random values from a uniform distribution
# Create two tensors

t1 = Tensor([[0, 1, 2], [4, 5, 6]], dtype=dtypes.float)
t2 = Tensor([[7, 8, 9], [10, 11, 12]], dtype=dtypes.float)

# t1 = Tensor([[0, 1, 2]], dtype=dtypes.int4)
# t2 = Tensor([[3, 3, 3]], dtype=dtypes.int4)
t6 = t1 + t2 

print(t6.numpy())

