import numpy as np
import torch


embed = torch.nn.Embedding(num_embeddings=4, embedding_dim=2)
embed.weight.data = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])

x = torch.tensor(
    [1, 0, 2],
    dtype=torch.int32,
)
out = embed(x)
print('Output:', np.array(np.nditer(out.detach().numpy(), order='F')))
gradient = torch.tensor(
    np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.6]).reshape(3, 2, order='F'),
    dtype=torch.float32,
)
out.backward(gradient)
print('dw: ', np.array(np.nditer(embed.weight.grad, order='F')))


def positional_encoding(sequence_length: int, model_dimension: int):
    pe = torch.zeros(sequence_length * model_dimension).reshape(sequence_length, model_dimension)
    for k in torch.arange(sequence_length):
        for i in torch.arange(model_dimension // 2):
            theta = k / (10_000 ** ((2 * i) / model_dimension))
            pe[k, 2 * i] = torch.sin(theta)
            pe[k, 2 * i + 1] = torch.cos(theta)
    return pe


embed = torch.nn.Embedding(num_embeddings=5, embedding_dim=4)
embed.weight.data = torch.tensor([
    [0.1, 0.1, 0.1, 0.1],
    [0.3, 0.3, 0.3, 0.3],
    [0.5, 0.5, 0.5, 0.5],
    [0.7, 0.7, 0.7, 0.7],
    [0.2, 0.2, 0.2, 0.2],
])

out = embed(x)
pe = positional_encoding(3, 4)
print('Output (positional): ', np.array(np.nditer(out.detach() + pe, order='F')))
