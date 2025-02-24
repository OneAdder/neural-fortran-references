import numpy as np
import torch


def create_fc(activation: torch.nn.Module) -> torch.nn.Sequential:
    fc_layer = torch.nn.Sequential(
        torch.nn.Linear(4, 5),
        activation,
        torch.nn.Linear(5, 4),
    )
    fc_layer._modules['0'].weight.data = torch.tensor(
        np.array(
            [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5]
        ).reshape(5, 4, order='F'),
        dtype=torch.float32,
    )
    fc_layer._modules['0'].bias.data = torch.zeros(5) + 0.11
    fc_layer._modules['2'].weight.data = torch.zeros(4, 5) + 0.1
    fc_layer._modules['2'].bias.data = torch.zeros(4) + 0.11
    return fc_layer


for activation in (torch.nn.ReLU(), torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.Softplus()):
    fc_layer = create_fc(activation)
    print('-------------------')
    print(f'Activation: {activation._get_name()}')
    print('-------------------')

    optim = torch.optim.SGD(fc_layer.parameters(), lr=0.01)

    x = torch.tensor(
        np.array(
            [0.0, -10.1, 0.2, 10.3, 0.4, 10.5, -0.6, 10.7, 10.8, 0.9, 0.11, 0.12],
        ).reshape(3, 4, order='F'),
        dtype=torch.float32,
        requires_grad=True,
    )
    out = fc_layer.forward(x)
    print('Output:', np.array(np.nditer(out.detach().numpy(), order='F')))
    out.backward(torch.tensor(
        np.array([0.1, 3., 2., 0.1, 3., 3., 0.1, 2., 0.1, 3., 0.1, 3.]).reshape(3, 4, order='F'),
        requires_grad=True,
    ))
    print(x.grad)
    print('Gradient:', np.array(np.nditer(x.grad, order='F')))

    optim.step()
    out = fc_layer.forward(x)
    print('Output after one step (SGD):', np.array(np.nditer(out.detach().numpy(), order='F')))
