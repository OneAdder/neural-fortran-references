import numpy as np
import torch


mha = torch.nn.MultiheadAttention(num_heads=2, embed_dim=4, batch_first=True)
mha.in_proj_weight.data = torch.zeros(12, 4) + 0.1
mha.in_proj_bias.data = torch.zeros(12) + 0.11
mha.out_proj.weight.data = torch.zeros(4, 4) + 0.1
mha.out_proj.bias.data = torch.zeros(4) + 0.11

optim = torch.optim.SGD(mha.parameters(), lr=0.01)

x = torch.tensor(
    np.array(
        [0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12],
    ).reshape(3, 4, 1, order='F').transpose(2, 0, 1),
    dtype=torch.float32,
    requires_grad=True,
)
out, weights = mha(x, x, x)
print('Output:', np.array(np.nditer(out.detach().numpy(), order='F')))
print('Attention Weights:', np.array(np.nditer(weights.detach().numpy(), order='F')))
gradient = torch.tensor(
    [.1, .1, .1, 3., 3., 3., 2., .1, 2., 3., .1, 3.],
    requires_grad=True,
).reshape(1, 3, 4)
out.backward(gradient=gradient)
print('Gradient:', np.array(np.nditer(x.grad.numpy(), order='F')))
optim.step()
out, weights = mha(x, x, x)
print('Output after one step (SGD):', np.array(np.nditer(out.detach(), order='F')))
