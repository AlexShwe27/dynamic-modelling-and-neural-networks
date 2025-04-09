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


class baseline_net(nn.Module):

    def __init__(self):
        super(baseline_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
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

def pendulum_baseline_neuralODE(t, state):

    state = state.requires_grad_(True)

    # Obtain the state derivatives directly from the neural network
    dstate = neural_network(state)

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
niters = 2000

neural_network = baseline_net()
optimizer = optim.Adam(neural_network.parameters(), lr=0.02)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
loss_function = nn.MSELoss()

# Set up lists for training and validation loss
train_loss = []
val_loss = []

for itr in range(1, niters + 1):

    epoch_loss = []

    mini_batches, mini_batches_y0 = get_batch(train_y)
    for batch_y, batch_y0 in zip(mini_batches, mini_batches_y0):

        pred = odeint(pendulum_baseline_neuralODE, batch_y0, batch_t, method="heun3")
        loss = loss_function(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    scheduler.step()

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    train_loss.append(avg_epoch_loss)

    # validation
    if itr % test_freq == 0:

        test_pred = odeint(pendulum_baseline_neuralODE, test_y0, test_t, method="heun3")
        test_loss = loss_function(test_pred, test_y)
        val_loss.append(test_loss.item())

        print('Training Process : {:.2f}% | Train Loss : {:.12f} | Test Loss : {:.12f} | Learning rate : {}'
              .format(itr / niters * 100, avg_epoch_loss, test_loss.item(), scheduler.get_last_lr()))


# Save the learning curve data
learning_data = pd.DataFrame({
    'Train Loss': train_loss,
    'Test Loss': [loss for loss in val_loss for _ in range(10)],
})

learning_data.to_csv("baseline_learning_Data.csv", index=False)

# Save the trained model
torch.save(neural_network.state_dict(), "pendulum_baseline_NN.pth")
