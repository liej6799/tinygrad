from tinygrad import Tensor
# rand = Tensor.rand(2, 3) # create a tensor of shape (2, 3) filled with random values from a uniform distribution
# Create two tensors

# t1 = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2D tensor (matrix)
# t2 = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2D tensor (matrix)

t1 = Tensor([[1, 2, 3], [4, 5, 6]])  # 2D tensor (matrix)
t2 = Tensor([[1, 2, 3], [4, 5, 6]])  # 2D tensor (matrix)


t6 = (t1 + t2)
print(t6.numpy())
