from tinygrad import Tensor
# rand = Tensor.rand(2, 3) # create a tensor of shape (2, 3) filled with random values from a uniform distribution
t4 = Tensor([1, 2, 3, 4, 5])
t5 = Tensor([6, 7, 8, 9, 10])
# t5 = (t4 + 1) * 2
# t6 = (t5 * t4).relu().log_softmax()
# print(t6.numpy())
print((t4 * t5).relu().log_softmax().numpy())