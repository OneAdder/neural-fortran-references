import numpy as np
import torch


mha = torch.nn.MultiheadAttention(num_heads=1, embed_dim=3, batch_first=True)
mha.in_proj_weight.data = torch.zeros(9, 3) + 0.1
mha.in_proj_bias.data = torch.zeros(9) + 0.11
mha.out_proj.weight.data = torch.zeros(3, 3) + 0.1
mha.out_proj.bias.data = torch.zeros(3) + 0.11

optim = torch.optim.SGD(mha.parameters(), lr=0.01)

query = torch.tensor(
    np.array(
        [-1., 0., 17., .4, 5., .6],
    ).reshape(2, 3, 1, order='F').transpose(2, 0, 1),
    dtype=torch.float32,
    requires_grad=True,
)
key_value = torch.tensor(
    np.array(
        [0.1, -.2, 0.3, 4., 15., 0.5],
    ).reshape(2, 3, 1, order='F').transpose(2, 0, 1),
    dtype=torch.float32,
    requires_grad=True,
)
out, weights = mha(query, key_value, key_value)
print('Output:', np.array(np.nditer(out.detach().numpy(), order='F')))
print('Attention Weights:', np.array(np.nditer(weights.detach().numpy(), order='F')))
gradient = torch.tensor(
    np.array([1., 2., .17, 4., .5, 6.]).reshape(2, 3, 1, order='F').transpose(2, 0, 1),
    requires_grad=True,
    dtype=torch.float32,
)
out.backward(gradient=gradient)
print('Query Gradient:', np.array(np.nditer(query.grad.numpy(), order='F')))
print('Key-Value Gradient:', np.array(np.nditer(key_value.grad.numpy(), order='F')))
