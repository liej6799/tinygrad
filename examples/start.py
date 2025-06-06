from tinygrad import Tensor
# rand = Tensor.rand(2, 3) # create a tensor of shape (2, 3) filled with random values from a uniform distribution
# Create two tensors
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[5, 6], [7, 8]])

# Add the tensors
c = a - b

# Print the result
print(c.numpy())