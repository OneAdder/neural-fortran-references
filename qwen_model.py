import numpy as np
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2Model


model = Qwen2Model(Qwen2Config(
    num_attention_heads=4,
    num_key_value_heads=2,
    hidden_size=8,
    intermediate_size=32,
    num_hidden_layers=2,
))
model.embed_tokens.weight.data = torch.tensor([-2., -0.223, 0.3, 3., -0.1, 1.28, 2.28, -2.28]).unsqueeze(0).expand(151936, -1)

model.layers[0].self_attn.q_proj.weight.data = torch.zeros(8, 8) + 0.1
model.layers[0].self_attn.q_proj.bias.data = torch.zeros(8) + 0.11
model.layers[0].self_attn.k_proj.weight.data = torch.zeros(4, 8) + 0.2
model.layers[0].self_attn.k_proj.bias.data = torch.zeros(4) + 0.11
model.layers[0].self_attn.v_proj.weight.data = torch.zeros(4, 8) + 0.3
model.layers[0].self_attn.v_proj.bias.data = torch.zeros(4) + 0.11
model.layers[0].self_attn.o_proj.weight.data = torch.zeros(8, 8) + 0.2
model.layers[0].mlp.gate_proj.weight.data = torch.zeros(32, 8) + 0.01
model.layers[0].mlp.up_proj.weight.data = torch.zeros(32, 8) + 0.05
model.layers[0].mlp.down_proj.weight.data = torch.zeros(8, 32) + 0.1

model.layers[1].self_attn.q_proj.weight.data = torch.zeros(8, 8) + 0.2
model.layers[1].self_attn.q_proj.bias.data = torch.zeros(8) + 0.11
model.layers[1].self_attn.k_proj.weight.data = torch.zeros(4, 8) + 0.1
model.layers[1].self_attn.k_proj.bias.data = torch.zeros(4) + 0.11
model.layers[1].self_attn.v_proj.weight.data = torch.zeros(4, 8) + 0.3
model.layers[1].self_attn.v_proj.bias.data = torch.zeros(4) + 0.11
model.layers[1].self_attn.o_proj.weight.data = torch.zeros(8, 8) + 0.1
model.layers[1].mlp.gate_proj.weight.data = torch.zeros(32, 8) + 0.02
model.layers[1].mlp.up_proj.weight.data = torch.zeros(32, 8) + 0.06
model.layers[1].mlp.down_proj.weight.data = torch.zeros(8, 32) + 0.2


input_ids = torch.LongTensor([[641, 9881, 358, 653, 537, 2948, 39244, 448, 10485]])
attention_mask = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])

out = model.forward(input_ids=input_ids, attention_mask=attention_mask)
print('Output:', np.array(np.nditer(out.last_hidden_state.detach().numpy().transpose(1, 2, 0), order='F')))
