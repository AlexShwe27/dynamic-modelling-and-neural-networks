import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torchdiffeq import odeint
import pandas as pd

# ===================================================================================================================
# DATA PREPARATION
# ===================================================================================================================

# -------------------------------------------------------------------------------------------------------------------
# Load True Data
# -------------------------------------------------------------------------------------------------------------------

# Load simulation data
sim_data = pd.read_csv('cartpole_simulation_data.csv')
sim_data = sim_data.iloc[:-1]
sim_data = sim_data.iloc[::10].reset_index(drop=True)

# Extract the variables from the DataFrame
time = sim_data.iloc[:, 0].to_numpy(dtype=np.float32)
angle = sim_data.iloc[:, 1].to_numpy(dtype=np.float32)
position = sim_data.iloc[:, 2].to_numpy(dtype=np.float32)
angular_momentum = sim_data.iloc[:, 3].to_numpy(dtype=np.float32)
linear_momentum = sim_data.iloc[:, 4].to_numpy(dtype=np.float32)

# Time
time = torch.tensor(time, dtype=torch.float32)

# True Trajectory
true_trajectory = np.column_stack((angle, position, angular_momentum, linear_momentum))
true_trajectory = torch.tensor(true_trajectory, dtype=torch.float32)

# Initial Conditions
init_cond = true_trajectory[0, :]

# -------------------------------------------------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------------------------------------------------
# Data parameters
data_size = len(sim_data)
batch_count = 16
batch_time = 10

# Simulation time
nsteps = data_size


# ===================================================================================================================
# NEURAL ORDINARY DIFFERENTIAL EQUATION
# ===================================================================================================================

# -------------------------------------------------------------------------------------------------------------------
# Neural Networks
# -------------------------------------------------------------------------------------------------------------------

class factorised_kinetic_energy_net(nn.Module):

    def __init__(self):
        super(factorised_kinetic_energy_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 384),
            nn.Tanh(),
            nn.Linear(384, 384),
            nn.Tanh(),
            nn.Linear(384, 3)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, state):
        return self.net(state)


class potential_energy_net(nn.Module):

    def __init__(self):
        super(potential_energy_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 192),
            nn.Tanh(),
            nn.Linear(192, 192),
            nn.Tanh(),
            nn.Linear(192, 1)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, state):
        return self.net(state)


# -------------------------------------------------------------------------------------------------------------------
# Neural ODE
# -------------------------------------------------------------------------------------------------------------------

def factorised_splitenergy_neuralODE(t, state):

    # Ensure state has gradients enabled
    state = state.requires_grad_(True)

    # Extract q and p from state considering it might be batched
    if state.dim() == 2:
        q = state[:, :2]
        p = state[:, 2:]
        theta = state[:, 0].unsqueeze(1)
    else:
        q = state[:2].unsqueeze(0)
        p = state[2:].unsqueeze(0)
        theta = state[0].unsqueeze(0)

    # Obtain the components of the lower triangular matrix from neural network
    A_components = factorised_kinetic_neural_network(q)
    A_11 = A_components[:, 0]
    A_21 = A_components[:, 1]
    A_22 = A_components[:, 2]

    # Construct inverse mass matrix
    A = torch.zeros(q.shape[0], 2, 2)
    A[:, 0, 0] = A_11
    A[:, 1, 0] = A_21
    A[:, 1, 1] = A_22

    # Compute the conjugate of A
    A_conj = A.transpose(-2, -1)

    # Compute the inverse mass matrix
    M_inv = torch.bmm(A, A_conj)

    # Compute the components of A Jacobian w.r.t theta
    dAdq_list = []
    dAdq_conj_list = []
    grad_M_inv_list = []
    dTdq_list = []

    for i in range(q.shape[1]):
        grad_00 = torch.autograd.grad(A[:, 0, 0], q, torch.ones_like(A[:, 0, 0]), create_graph=True)[0][:, i]
        grad_10 = torch.autograd.grad(A[:, 1, 0], q, torch.ones_like(A[:, 1, 0]), create_graph=True)[0][:, i]
        grad_11 = torch.autograd.grad(A[:, 1, 1], q, torch.ones_like(A[:, 1, 1]), create_graph=True)[0][:, i]

        zeros = torch.zeros_like(grad_00)

        dAdq_temp = torch.stack([
            torch.stack([grad_00, zeros], dim=1),
            torch.stack([grad_10, grad_11], dim=1)
        ], dim=1)

        dAdq_conj_temp = dAdq_temp.transpose(-2, -1)
        grad_M_inv_temp = torch.bmm(dAdq_temp, A_conj) + torch.bmm(A, dAdq_conj_temp)
        dTdq_temp = 0.5 * torch.bmm(p.unsqueeze(1), torch.bmm(grad_M_inv_temp, p.unsqueeze(-1))).squeeze()

        dAdq_list.append(dAdq_temp)
        dAdq_conj_list.append(dAdq_conj_temp)
        grad_M_inv_list.append(grad_M_inv_temp)
        dTdq_list.append(dTdq_temp)

    # Concatenate after loop
    dAdq = torch.stack(dAdq_list, dim=1)
    dAdq_conj = torch.stack(dAdq_conj_list, dim=1)
    grad_M_inv = torch.stack(grad_M_inv_list, dim=1)
    if state.dim() == 2:
        dTdq = torch.stack(dTdq_list, dim=1)
    else:
        dTdq = torch.stack(dTdq_list)

    dTdp = torch.bmm(M_inv, p.unsqueeze(-1)).squeeze()

    V = potential_neural_network(q)
    dVdq = torch.autograd.grad(V, q, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    if state.dim() == 2:
        dqdt = dTdp
        dpdt = -dTdq - dVdq

        dstate = torch.cat([dqdt, dpdt], dim=1)
    else:
        dqdt = dTdp
        dpdt = -dTdq - dVdq.squeeze()

        dstate = torch.cat([dqdt, dpdt])

    return dstate


# ===================================================================================================================
# TRAINING / FITTING
# ===================================================================================================================

# -------------------------------------------------------------------------------------------------------------------
# Train Data
# -------------------------------------------------------------------------------------------------------------------

# Slice train and test data in 80 and 20 percentages
split_idx = int(len(time) * 0.8)
train_size = data_size * 0.8

train_t = time[:split_idx]
train_y = true_trajectory[:split_idx, :]
train_y = torch.chunk(train_y, int(train_size/batch_time))
train_y0 = init_cond

batch_t = train_t[:batch_time]


def get_batch(train_data):

    shuffled_trained_data = torch.stack(tuple(train_data[i] for i in torch.randperm(len(train_data))))

    mini_batches = torch.chunk(shuffled_trained_data, batch_count)
    mini_batches = tuple(tensor.permute(1, 0, 2) for tensor in mini_batches)
    mini_batches_y0 = tuple(tensor[0, :, :] for tensor in mini_batches)

    return mini_batches, mini_batches_y0


# -------------------------------------------------------------------------------------------------------------------
# Test Data
# -------------------------------------------------------------------------------------------------------------------

test_t = time[split_idx:]
test_y = true_trajectory[split_idx:, :]
test_y0 = test_y[0, :]

# -------------------------------------------------------------------------------------------------------------------
# Train Loop
# -------------------------------------------------------------------------------------------------------------------

test_freq = 10
niters = 10000

factorised_kinetic_neural_network = factorised_kinetic_energy_net()
optimiser = optim.Adam(factorised_kinetic_neural_network.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=500, gamma=0.5)

potential_neural_network = potential_energy_net()
potential_neural_network.load_state_dict(torch.load("cartpole_potential_energy_NN.pth", weights_only=True))
potential_neural_network.eval()

loss_function = nn.MSELoss()

# Set up lists for training and validation loss
train_loss = []
val_loss = []

for itr in range(1, niters + 1):

    epoch_loss = []

    mini_batches, mini_batches_y0 = get_batch(train_y)
    for batch_y, batch_y0 in zip(mini_batches, mini_batches_y0):
        pred = odeint(factorised_splitenergy_neuralODE, batch_y0, batch_t, method="heun3")
        loss = loss_function(pred, batch_y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch_loss.append(loss.item())

    scheduler.step()
    scheduler.step()

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    train_loss.append(avg_epoch_loss)

    # validation
    if itr % test_freq == 0:

        test_pred = odeint(factorised_splitenergy_neuralODE, test_y0, test_t, method="heun3")
        test_loss = loss_function(test_pred, test_y)
        val_loss.append(test_loss.item())

        print('Training Process : {:.2f}% | Train Loss : {:.16f} | Test Loss : {:.16f} | Learning rate : {}'
              .format(itr / niters * 100, avg_epoch_loss, test_loss.item(), scheduler.get_last_lr()))

# Save the learning curve data
learning_data = pd.DataFrame({
    'Train Loss': train_loss,
    'Test Loss': [loss for loss in val_loss for _ in range(10)]
})

learning_data.to_csv("Learning_Data.csv", index=False)

# Save the trained model
torch.save(factorised_kinetic_neural_network.state_dict(), "cartpole_factorised_kinetic_energy_NN.pth")
