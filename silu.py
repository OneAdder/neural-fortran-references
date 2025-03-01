import torch


silu = torch.nn.SiLU()

x = torch.tensor([-3., -0.5, 0.1, 15.], requires_grad=True)
gradient = torch.tensor([1., 2., 0.5, 3.])
out = silu(x)
print('Output:', out)
out.backward(gradient)
print('Gradient:', x.grad)
