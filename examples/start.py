from tinygrad import Tensor
rand = Tensor.rand(2, 3) # create a tensor of shape (2, 3) filled with random values from a uniform distribution
print(rand.numpy())