import numpy as np
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2DecoderLayer


decoder = Qwen2DecoderLayer(Qwen2Config(num_attention_heads=4, num_key_value_heads=2, hidden_size=8, intermediate_size=32), 0)

decoder.self_attn.q_proj.weight.data = torch.zeros(8, 8) + 0.1
decoder.self_attn.q_proj.bias.data = torch.zeros(8) + 0.11
decoder.self_attn.k_proj.weight.data = torch.zeros(4, 8) + 0.2
decoder.self_attn.k_proj.bias.data = torch.zeros(4) + 0.11
decoder.self_attn.v_proj.weight.data = torch.zeros(4, 8) + 0.3
decoder.self_attn.v_proj.bias.data = torch.zeros(4) + 0.11
decoder.self_attn.o_proj.weight.data = torch.zeros(8, 8) + 0.2

decoder.mlp.gate_proj.weight.data = torch.zeros(32, 8) + 0.01
decoder.mlp.up_proj.weight.data = torch.zeros(32, 8) + 0.05
decoder.mlp.down_proj.weight.data = torch.zeros(8, 32) + 0.1


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
cos = torch.tensor(
    [[
        [1.0000, 1.0000],
        [ 0.5403, 0.5403],
        [-0.4161, -0.4161],
        [-0.9900, -0.9900],
        [-0.6536, -0.6536],
        [ 0.2837, 0.2837],
        [ 0.9602, 0.9602],
        [ 0.7539, 0.7539],
        [-0.1455, -0.1455],
    ]],
    dtype=torch.float32,
)
sin = torch.tensor(
    [[
        [0.0000, 0.0000],
        [0.8415, 0.8415],
        [0.9093, 0.9093],
        [0.1411, 0.1411],
        [-0.7568, -0.7568],
        [-0.9589, -0.9589],
        [-0.2794, -0.2794],
        [0.6570, 0.6570],
        [0.9894, 0.9894],
    ]]
)

decoder.self_attn.config._attn_implementation = 'sdpa'
out, = decoder.forward(x, position_embeddings=(cos, sin), attention_mask=None)
print('Output:', np.array(np.nditer(out.detach().numpy(), order='F')))

gradient = torch.tensor([[
    [0.2643, 0.4271, 0.8704, 0.7629, 0.5051, 0.6396, 0.5826, 0.7600],
    [0.5053, 0.8181, 0.5688, 0.1311, 0.2189, 0.1247, 0.6509, 0.9715],
    [0.8736, 0.9274, 0.6746, 0.7574, 0.4260, 0.9805, 0.1613, 0.5208],
    [0.7107, 0.3118, 0.9214, 0.9603, 0.4310, 0.7546, 0.4260, 0.9790],
    [0.4371, 0.4004, 0.3114, 0.6012, 0.3824, 0.5645, 0.8945, 0.1589],
    [0.7562, 0.7692, 0.9759, 0.3955, 0.7647, 0.9313, 0.3592, 0.6575],
    [0.4131, 0.8253, 0.6638, 0.4760, 0.7132, 0.4169, 0.5082, 0.1034],
    [0.2426, 0.8845, 0.0278, 0.9791, 0.1745, 0.7757, 0.8625, 0.0686],
    [0.3533, 0.5925, 0.3023, 0.9799, 0.3256, 0.0203, 0.3525, 0.5525],
]], dtype=torch.float32)
out.backward(gradient)
print('Gradient:', np.array(np.nditer(x.grad, order='F')))
