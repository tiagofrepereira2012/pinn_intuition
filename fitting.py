import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Balistic(Dataset):
    def __init__(self, s_x, s_y, t, t_phys):
        self.s_x = s_x.astype(np.float32)
        self.s_y = s_y.astype(np.float32)
        self.t = t.astype(np.float32)
        self.t_phys = t_phys.astype(np.float32)
        self.physics_indexes = len(self.t_phys)

    def __len__(self):
        return len(self.s_x)

    def __getitem__(self, idx):
        return (
            self.s_x[idx],
            self.s_y[idx],
            self.t[idx],
            self.t_phys[np.random.random_integers(0, self.physics_indexes - 1)],
        )


def simple_nn_train_loop(model, epochs, data_loader, optimizer, criterion, phys_weight=None):
    # Overfit the model
    for _ in range(epochs):
        model.train()
        for s_x_, s_y_, t_, _ in data_loader:
            optimizer.zero_grad()
            s_hat = model(t_)
            loss = criterion(torch.unsqueeze(s_hat[:, 0], 1), s_x_) + criterion(torch.unsqueeze(s_hat[:, 1], 1), s_y_)
            loss.backward()
            optimizer.step()

        # print(f"Epoch {epoch}, validation loss: {loss:.2f}")
    return model


def simple_pinn_train_loop(model, epochs, data_loader, optimizer, criterion, phys_weight=0.2):
    # Overfit the model
    mu = 0.0
    g = 9.81
    mu = torch.tensor([mu, mu], dtype=torch.float32, requires_grad=True)
    # mu = torch.tensor(mu, dtype=torch.float32, requires_grad=True)

    for _ in range(epochs):

        model.train()
        for s_x_, s_y_, t_, t_phys in data_loader:
            optimizer.zero_grad()
            t_phys.requires_grad = True
            s_hat = model(t_)
            loss = criterion(torch.unsqueeze(s_hat[:, 0], 1), s_x_) + criterion(torch.unsqueeze(s_hat[:, 1], 1), s_y_)

            s_hat_phys = model(t_phys)
            s_hat_phys_x = torch.unsqueeze(s_hat_phys[:, 0], 1)
            s_hat_phys_y = torch.unsqueeze(s_hat_phys[:, 1], 1)

            ds_dt_x = torch.autograd.grad(
                s_hat_phys_x,
                t_phys,
                grad_outputs=torch.ones_like(s_hat_phys_x),
                retain_graph=True,
                allow_unused=True,
                create_graph=True,
            )[0]

            ds_dt_y = torch.autograd.grad(
                s_hat_phys_y,
                t_phys,
                grad_outputs=torch.ones_like(s_hat_phys_x),
                retain_graph=True,
                allow_unused=True,
                create_graph=True,
            )[0]

            # ds_dt.requires_grad = True
            dv_dt_x = torch.autograd.grad(
                ds_dt_x,
                t_phys,
                retain_graph=True,
                grad_outputs=torch.ones_like(ds_dt_x),
            )[0]

            dv_dt_y = torch.autograd.grad(
                ds_dt_y,
                t_phys,
                retain_graph=True,
                grad_outputs=torch.ones_like(ds_dt_y),
            )[0]

            # import ipdb

            # ipdb.set_trace()
            ds_dt = torch.cat((ds_dt_x, ds_dt_y), dim=1)
            dv_dt = torch.cat((dv_dt_x, dv_dt_y), dim=1)
            ds_dt.detach()
            dv_dt.detach()

            g_ = torch.tensor([0.0, g], dtype=torch.float32)

            loss_phys = torch.mean(torch.square(-mu * torch.nn.functional.normalize(ds_dt, dim=1) * ds_dt - g_ - dv_dt))

            loss = loss + phys_weight * loss_phys
            # print(f"{loss} - {loss_phys} - {mu}")

            # import ipdb

            # ipdb.set_trace()

            loss.backward()
            # Improvised learning rate for the drag force
            optimizer.step()
            mu.data = mu.data - 0.01 * mu.grad.data
            mu.grad.data.zero_()

            pass

        # print(f"Epoch {epoch}, validation loss: {loss:.2f}")

    return model


def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def train_nn(s_x, s_y, t, model, epochs=10, data_offset=None, train_fn=simple_nn_train_loop, phys_weight=0.2):
    """
    Train a simple neural network.
    """

    # Sampling the time on the whole domain
    t_phys = np.array(t)[np.random.random_integers(0, len(t) - 1, size=len(t))]
    t_phys = np.expand_dims(t_phys, axis=1)

    x_train = np.expand_dims(np.array(s_x if data_offset is None else s_x[0:data_offset]), axis=1)
    y_train = np.expand_dims(np.array(s_y if data_offset is None else s_y[0:data_offset]), axis=1)
    t_train = np.expand_dims(np.array(t if data_offset is None else t[0:data_offset]), axis=1)

    # Normalizing
    mu_x = np.mean(x_train)
    std_x = np.std(x_train)
    mu_y = np.mean(y_train)
    std_y = np.std(y_train)

    x_train = (x_train - mu_x) / std_x
    y_train = (y_train - mu_y) / std_y

    data_loader = torch.utils.data.DataLoader(
        dataset=Balistic(x_train, y_train, t_train, t_phys),
        batch_size=32,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01, momentum=0.9)
    criterion = torch.nn.MSELoss()

    model = train_fn(model, epochs, data_loader, optimizer, criterion, phys_weight=phys_weight)

    model.eval()
    with torch.no_grad():
        s_hat = model(torch.Tensor(np.expand_dims(np.array(t, dtype=np.float32), axis=1)))

    s_hat = s_hat.detach().numpy()

    s_hat[:, 0] = s_hat[:, 0] * std_x + mu_x
    s_hat[:, 1] = s_hat[:, 1] * std_y + mu_y

    return s_hat


"""
from data_generator import (
    perfect_balistic_data_with_air_resistance,
    noisy_balistic_data_with_air_resistance,
)

s_x, s_y, t = perfect_balistic_data_with_air_resistance(
    500, 45, mass=10.43, rho=1.2, dt=0.1, Cd=0.47, sphere_radius=0.11
)

# Create a simple neural network
layers = []
layers.append(nn.Linear(1, 128))
layers.append(nn.GELU())
layers.append(nn.Linear(128, 2))
model = nn.Sequential(*layers)
print(model)


s_hat = train_nn(s_x, s_y, t, model=model, epochs=10, train_fn=simple_pinn_train_loop)
"""
