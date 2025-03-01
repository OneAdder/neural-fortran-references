import numpy as np
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2MLP


feed_forward = Qwen2MLP(Qwen2Config(hidden_act='silu', intermediate_size=4, hidden_size=3))
feed_forward.gate_proj.weight.data = torch.tensor([[0.1, 0.2, 0.3, 0.4]] * 3).transpose(-2, 1)
feed_forward.up_proj.weight.data = torch.tensor([[0.7, 0.6, 0.5, 0.8]] * 3).transpose(-2, 1)
feed_forward.down_proj.weight.data = torch.tensor([[0.15, 0.25, 0.35]] * 4).transpose(-2, 1)

x = torch.tensor(
    np.array(
        [-0.9812, -2., -1.0309, -0.9520, 1.0083, -1.0007],
    ).reshape(2, 3, order='F'),
    dtype=torch.float32,
    requires_grad=True,
)

out = feed_forward.forward(x)
print('Output:', np.array(np.nditer(out.detach().numpy(), order='F')))
out.backward(torch.tensor(
    np.array([0.1, 3., 2., 0.3, 3.1, 0.1]).reshape(2, 3, order='F'),
    requires_grad=True,
))
print('Gradient:', np.array(np.nditer(x.grad, order='F')))
print('`gate_proj` weights gradient:', np.array(np.nditer(feed_forward.gate_proj.weight.grad.T, order='F')))
print('`up_proj` weights gradient:', np.array(np.nditer(feed_forward.up_proj.weight.grad.T, order='F')))
print('`down_proj` weights gradient:', np.array(np.nditer(feed_forward.down_proj.weight.grad.T, order='F')))
