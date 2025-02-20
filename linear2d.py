import numpy as np
import torch


inp = torch.tensor(
    np.array([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12]).reshape(3, 4, order='F')
)
inp = torch.tensor(inp, requires_grad=True, dtype=torch.float32)

linear = torch.nn.Linear(in_features=4, out_features=1)
linear.bias.data = torch.tensor([0.11, 0.11, 0.11, 0.11])
linear.weight.data = torch.zeros(4, 4) + 0.1

result = linear.forward(inp)
print('Output: ', np.array(np.nditer(result.detach(), order='F')))

result.backward(
    torch.tensor(
        np.array([0.1, 3., 2., 0.1, 3., 3., 0.1, 2., 0.1, 3., 0.1, 3.]).reshape(3, 4, order='F'),
        dtype=torch.float32,
    )
)
print('Gradient: ', np.array(np.nditer(inp.detach(), order='F')))
print('dw: ', np.array(np.nditer(linear.weight.grad, order='F')))
print('db: ', np.array(np.nditer(linear.bias.grad, order='F')))
