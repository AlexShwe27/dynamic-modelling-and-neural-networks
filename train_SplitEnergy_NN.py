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
sim_data = pd.read_csv('pendulum_simulation_data.csv')
sim_data = sim_data.iloc[:-1]
sim_data = sim_data.iloc[::10].reset_index(drop=True)

# Extract the variables from the DataFrame
time = sim_data.iloc[:, 0].to_numpy(dtype=np.float32)
angle = sim_data.iloc[:, 1].to_numpy(dtype=np.float32)
angular_momentum = sim_data.iloc[:, 2].to_numpy(dtype=np.float32)

# Time
time = torch.tensor(time, dtype=torch.float32)

# True Trajectory
true_trajectory = np.column_stack((angle, angular_momentum))
true_trajectory = torch.tensor(true_trajectory, dtype=torch.float32)

# Initial Conditions
init_cond = true_trajectory[0, :]

# -------------------------------------------------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------------------------------------------------
# Data parameters
data_size = len(sim_data)
batch_count = 8
batch_time = 10

# Simulation time
nsteps = data_size

# ===================================================================================================================
# NEURAL ORDINARY DIFFERENTIAL EQUATION
# ===================================================================================================================

# -------------------------------------------------------------------------------------------------------------------
# Neural Network
# -------------------------------------------------------------------------------------------------------------------


class kinetic_energy_net(nn.Module):

    def __init__(self):
        super(kinetic_energy_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
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
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
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

def pendulum_splitenergy_neuralODE(t, state):

    state = state.requires_grad_(True)

    if state.dim() == 2:
        q = state[:, 0].unsqueeze(1)
        p = state[:, 1].unsqueeze(1)
    else:
        q = state[0].unsqueeze(0)
        p = state[1].unsqueeze(0)

    # Obtain the kinetic energy and potential energy from Neural Network
    T = kinetic_neural_network(state)
    V = potential_neural_network(q)

    # Calculate the gradients of kinetic and potential energy w.r.t the respective states
    dT = torch.autograd.grad(outputs=T, inputs=state, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    dVdq = torch.autograd.grad(outputs=V, inputs=q, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    if state.dim() == 2:
        # Calculate dqdt and dpdt
        dqdt = dT[:, 1]
        dpdt = -dT[:, 0] - dVdq.squeeze()
        # Combine dqdt and dpdt into a single tensor
        dstate = torch.stack([dqdt, dpdt], dim=1)

    else:
        # Calculate dqdt and dpdt
        dqdt = dT[1]
        dpdt = -dT[0] - dVdq
        # Combine dqdt and dpdt into a single tensor
        dstate = torch.stack([dqdt, dpdt.squeeze()])

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

kinetic_neural_network = kinetic_energy_net()
kinetic_optimizer = optim.Adam(kinetic_neural_network.parameters(), lr=0.01)
kinetic_scheduler = optim.lr_scheduler.StepLR(kinetic_optimizer, step_size=500, gamma=0.5)

potential_neural_network = potential_energy_net()
potential_optimizer = optim.Adam(potential_neural_network.parameters(), lr=0.01)
potential_scheduler = optim.lr_scheduler.StepLR(potential_optimizer, step_size=500, gamma=0.5)

loss_function = nn.MSELoss()

# Set up lists for training and validation loss
train_loss = []
val_loss = []

for itr in range(1, niters + 1):

    epoch_loss = []

    mini_batches, mini_batches_y0 = get_batch(train_y)
    for batch_y, batch_y0 in zip(mini_batches, mini_batches_y0):

        pred = odeint(pendulum_splitenergy_neuralODE, batch_y0, batch_t, method="heun3")
        loss = loss_function(pred, batch_y)

        kinetic_optimizer.zero_grad()
        potential_optimizer.zero_grad()
        loss.backward()
        kinetic_optimizer.step()
        potential_optimizer.step()

        epoch_loss.append(loss.item())

    kinetic_scheduler.step()
    potential_scheduler.step()

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    train_loss.append(avg_epoch_loss)

    # validation
    if itr % test_freq == 0:

        test_pred = odeint(pendulum_splitenergy_neuralODE, test_y0, test_t, method="heun3")
        test_loss = loss_function(test_pred, test_y)
        val_loss.append(test_loss.item())

        print('Training Process : {:.2f}% | Train Loss : {:.16f} | Test Loss : {:.16f} | Learning rate : {}'
              .format(itr / niters * 100, avg_epoch_loss, test_loss.item(), kinetic_scheduler.get_last_lr()))


# Save the learning curve data
learning_data = pd.DataFrame({
    'Train Loss': train_loss,
    'Test Loss': [loss for loss in val_loss for _ in range(10)]
})

learning_data.to_csv("SENN_Learning_Data.csv", index=False)

# Save the trained model
# Save the trained models
torch.save(kinetic_neural_network.state_dict(), "pendulum_kinetic_energy_NN.pth")
torch.save(potential_neural_network.state_dict(), "pendulum_potential_energy_NN.pth")
