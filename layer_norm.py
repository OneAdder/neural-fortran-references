import numpy as np
import torch


layer_norm = torch.nn.LayerNorm(4)


x = torch.tensor(
    np.array(
        [0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12],
    ).reshape(3, 4, order='F'),
    dtype=torch.float32,
    requires_grad=True,
)
out = layer_norm.forward(x)
print('Output:', np.array(np.nditer(out.detach().numpy(), order='F')))
out.backward(torch.tensor(
    np.array([0.1, 3., 2., 0.1, 3., 3., 0.1, 2., 0.1, 3., 0.1, 3.]).reshape(3, 4, order='F'),
    requires_grad=True,
))
print('Gradient:', np.array(np.nditer(x.grad, order='F')))
print('d_gamma:', np.array(np.nditer(layer_norm.weight.grad, order='F')))
print('d_beta:', np.array(np.nditer(layer_norm.bias.grad, order='F')))
