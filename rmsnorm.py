import numpy as np
import torch


rmsnorm = torch.nn.RMSNorm(3)

x = torch.tensor(
    np.array(
        [-0.5, 0.1, 12., -13., 0.6, -0.7],
    ).reshape(2, 3, order='F'),
    dtype=torch.float32,
    requires_grad=True,
)
out = rmsnorm.forward(x)
print('Output:', np.array(np.nditer(out.detach().numpy(), order='F')))
out.backward(torch.tensor(
    np.array([0.1, 3.1, 2.7, 3., 1.4, 0.2]).reshape(2, 3, order='F'),
    requires_grad=True,
))
print('Gradient:', np.array(np.nditer(x.grad, order='F')))
print('d_gamma:', np.array(np.nditer(rmsnorm.weight.grad, order='F')))
