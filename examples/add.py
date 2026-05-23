from tinygrad import Tensor

if __name__ == "__main__":
  a = Tensor([1.0, 2.0, 3.0, 4.0])
  b = Tensor([10.0, 20.0, 30.0, 40.0])
  c = a + b
  print(f"a = {a.tolist()}")
  print(f"b = {b.tolist()}")
  print(f"a + b = {c.tolist()}")
