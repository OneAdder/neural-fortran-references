import numpy as np
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2RotaryEmbedding


rope = Qwen2RotaryEmbedding(Qwen2Config(num_attention_heads=1, num_key_value_heads=2, hidden_size=8))

x = torch.tensor(
    [[
            [0.3707, -0.9110, -1.1827,  1.2871, -0.4760, -0.2371,  1.1053, -1.5656],
            [1.3589,  0.0424, -0.8801,  0.1248, -1.2932,  0.0050,  1.0893, 1.5733],
            [1.9885, -0.4536,  1.0227,  0.1316,  1.0930,  1.2334, -0.1328, -0.1736],
            [-0.3016,  0.3615,  0.2538, -1.5641, -0.3046,  1.8437,  0.4158, 1.2495],
            [0.8139,  0.8447,  1.6236, -0.1454, -0.7212,  0.1448,  1.8390, 0.1076],
            [-0.8695,  0.5775,  1.6021, -0.5928,  1.3227,  1.4683, -0.2432, -0.0873],
            [-1.6688,  0.0844,  1.6466, -0.6204, -0.4072, -0.1733, -0.7129, -1.1768],
            [1.1663, -0.6990,  1.0529, -1.8024,  0.6711, -1.1278,  0.0029, -0.2305],
            [0.2469, -1.0675, -2.2763,  0.3605,  0.3306,  0.5619,  0.2979, -0.9782],
    ]],
    dtype=torch.float32,
    requires_grad=True,
)
position_ids = torch.arange(0, 9, dtype=torch.float32).view(1, 9)
cos, sin = rope(x, position_ids)
print('Cos:', np.array(np.nditer(cos.detach(), order='F')))
print('Sin:', np.array(np.nditer(sin.detach(), order='F')))
