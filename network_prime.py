import numpy as np
import random
import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

torch.manual_seed(0)

data_path = 'D:/machine_learning_data/SelfDriving/processed_data/gray_video_frames70x160'
reversed_data_path = 'D:/machine_learning_data/SelfDriving/processed_data/gray_reversed_video_frames70x160'
label_path = 'D:/machine_learning_data/SelfDriving/processed_data/adjusted_labels'
reversed_label_path = 'D:/machine_learning_data/SelfDriving/processed_data/adjusted_reversed_labels'

validation_path = 'D:/machine_learning_data/SelfDriving/processed_data/gray_validation_set70x160'


def get_validation_set():
    validation = os.listdir(validation_path)

    validation_data = []
    validation_labels = []

    for lab in validation:
        if lab[-18] == 'l' or lab[-17] == 'l':
            validation_labels.append(lab)
        else:
            validation_data.append(lab)

    return list(zip(validation_data, validation_labels))


def get_data_names(data_packs=None, shuffle=False):

    if data_packs is None:
        data_packs = ['all']

    dat_arr = os.listdir(data_path)
    lab_arr = os.listdir(label_path)

    org_dat_arr = []
    org_lab_arr = []
    new_dat_arr = []
    new_lab_arr = []
    uda_dat_arr = []
    uda_lab_arr = []
    ud3_dat_arr = []
    ud3_lab_arr = []

    for name in dat_arr:
        if name[0] == 'u':
            if name[1] == '3':
                ud3_dat_arr.append(name)
            else:
                uda_dat_arr.append(name)
        elif name[0] == 'g':
            org_dat_arr.append(name)
        elif name[0] == 'n':
            new_dat_arr.append(name)

    for name in lab_arr:
        if name[0] == 'u':
            if name[1] == '3':
                ud3_lab_arr.append(name)
            else:
                uda_lab_arr.append(name)
        elif name[0] == 'l':
            org_lab_arr.append(name)
        elif name[0] == 'n':
            new_lab_arr.append(name)

    rev_dat_arr = os.listdir(reversed_data_path)
    rev_lab_arr = os.listdir(reversed_label_path)

    org_lab_arr.sort()
    org_dat_arr.sort()

    new_lab_arr.sort()
    new_dat_arr.sort()

    uda_lab_arr.sort()
    uda_dat_arr.sort()

    ud3_lab_arr.sort()
    ud3_dat_arr.sort()

    ud3_lab_arr_2 = []
    ud3_dat_arr_2 = []

    for i in range(len(ud3_lab_arr)):
        if i % 3 == 0:
            ud3_lab_arr_2.append(ud3_lab_arr[i])
            ud3_dat_arr_2.append(ud3_dat_arr[i])

    ud3_lab_arr = ud3_lab_arr_2
    ud3_dat_arr = ud3_dat_arr_2

    rev_lab_arr.sort(reverse=True)
    rev_dat_arr.sort(reverse=True)

    final_data, final_labels = [], []

    if ('all' in data_packs) or ('comma_new' in data_packs):
        for i in range(len(org_dat_arr)):
            final_data.append(org_dat_arr[i])
            final_labels.append(org_lab_arr[i])

    if ('all' in data_packs) or ('comma_new_rev' in data_packs):
        for i in range(len(rev_dat_arr)):
            final_data.append(rev_dat_arr[i])
            final_labels.append(rev_lab_arr[i])

    if ('all' in data_packs) or ('comma_old' in data_packs):
        for i in range(len(new_dat_arr)):
            final_data.append(new_dat_arr[i])
            final_labels.append(new_lab_arr[i])

    if ('all' in data_packs) or ('uda' in data_packs):
        for i in range(len(uda_dat_arr)):
            final_data.append(uda_dat_arr[i])
            final_labels.append(uda_lab_arr[i])

    if ('all' in data_packs) or ('ud3' in data_packs):
        for i in range(len(ud3_dat_arr)):
            final_data.append(ud3_dat_arr[i])
            final_labels.append(ud3_lab_arr[i])

    data_set = list(zip(final_data, final_labels))

    if shuffle is True:
        random.shuffle(data_set)

    return len(final_data), data_set


# Functions:
def graph_loss(train_losses, test_losses, g_time):
    fig, axs = plt.subplots(2)

    axs[0].plot(train_losses)
    axs[0].set_ylabel('Training Loss')

    axs[1].plot(test_losses, color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Validation Loss')

    if g_time > 1e6:
        plt.show(block=True)

    else:
        plt.show(block=False)
        plt.pause(g_time)
        plt.close()


# Class:
class Brain(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        # Equipment:
        # ================

        # Convolutional Layers:
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, bias=False)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=False)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, bias=False)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, bias=False)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        # Linear Layers:
        self.linear1 = nn.Linear(1664, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 10)
        self.linear4 = nn.Linear(10, 1)

        # Utility Layers:
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        # Batchnorm layers:
        self.batchnorm1 = nn.BatchNorm2d(24)
        self.batchnorm2 = nn.BatchNorm2d(36)
        self.batchnorm3 = nn.BatchNorm2d(48)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.batchnorm5 = nn.BatchNorm2d(64)

    def forward(self, x):
        # Modular Pass:
        # =-----------=
        # Conv pass
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)

        x = self.flatten(x)

        # Lin pass:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        return x


class PrimeControl:
    def __init__(self, brain_path=None, opt_path=None, lr=0.001, decay=0.01, in_channels=1):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.brain = Brain(in_channels=in_channels)
        self.brain.to(self.device)

        self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=lr, weight_decay=decay)
        self.loss = nn.MSELoss(reduction='sum')

        if brain_path is not None:
            self.load_brain(brain_path, opt_path)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

    def get_parameter_count(self):
        return sum([p.numel() for p in self.brain.parameters()])

    def load_brain(self, brain_path, train_dict_path=None):
        self.brain.load_state_dict(torch.load(os.path.join('models', brain_path)))
        if train_dict_path is not None:
            self.optimizer.load_state_dict(torch.load(os.path.join('models', train_dict_path)))
        return

    def save_brain(self, path, train_dict_path=None):
        print("Saving brain...")
        torch.save(self.brain.state_dict(), os.path.join('models', path))
        if train_dict_path is not None:
            torch.save(self.optimizer.state_dict(), os.path.join('models', train_dict_path))
        return

    def predict(self, frame, eval=True):
        # Set evaluation mode:
        if eval is True:
            self.brain.eval()
        frame = torch.from_numpy(frame).float().to(self.device)
        return self.brain(frame)

    def validation(self, batch_size=100, monitor_dynamics=False):
        # Right now, this calculates the average loss per prediction
        self.brain.eval()

        with torch.no_grad():
            validation_set = get_validation_set()

            current_average = 0.0
            nr_frames = 0.0

            for dat_name, lab_name in validation_set:
                val_X, val_y = np.load(os.path.join(validation_path, dat_name)), np.load(os.path.join(validation_path, lab_name))

                # Normalize data!
                val_X = (val_X/127.5) - 1.0
                # y to torch tensor:
                val_y = torch.from_numpy(val_y).float().to(self.device)

                batch_count = int(len(val_X)/batch_size)
                if len(val_X)/batch_size > batch_count:
                    batch_count += 1

                for i in range(batch_count):
                    val_batch_X = val_X[i*batch_size: min((i+1)*batch_size, len(val_X))]
                    val_batch_y = val_y[i*batch_size: min((i+1)*batch_size, len(val_X))]

                    predictions = self.predict(val_batch_X, eval=True).flatten()

                    if monitor_dynamics is True and i == 0:
                        for j in range(int(len(val_batch_y)/10)):
                            print(f"{predictions[j].item()}|{val_batch_y[j]}")

                    prediction_loss = self.loss(predictions, val_batch_y)

                    current_average = (nr_frames/(nr_frames + len(val_batch_X)))*current_average + prediction_loss.item()/(nr_frames + len(val_batch_X))
                    nr_frames += len(val_batch_X)

            return current_average

    def train(self, epochs=1, lr=None, decay=None, batch_size=50, save_mode='all', validation=True, verbose=2,
              graph_time=0, monitor_val_predictions=False, dat_packs=['all'], init_min_val_loss=1000):

        # Set training mode:
        self.brain.train()

        min_val_loss = init_min_val_loss

        print("Beginning training...")

        if validation is True:
            print(f"Initial validation loss : {self.validation(100, monitor_dynamics=monitor_val_predictions)}")
            self.brain.train()

        # Set learning rate:
        if lr is not None:
            for gr in self.optimizer.param_groups:
                gr['lr'] = lr
        if decay is not None:
            for gr in self.optimizer.param_groups:
                gr['weight_decay'] = decay

        epoch_train_losses = []
        epoch_test_losses = []

        for epoch in range(epochs):

            data_len, data_set = get_data_names(dat_packs, shuffle=True)

            current_average = 0.0
            nr_frames = 0.0

            for i, (vid_name, lab_name) in tqdm(enumerate(data_set), total=data_len):

                if vid_name[0] == 'r':
                    X, y = np.load(os.path.join(reversed_data_path, vid_name)), np.load(
                        os.path.join(reversed_label_path, lab_name))
                else:
                    X, y = np.load(os.path.join(data_path, vid_name)), np.load(os.path.join(label_path, lab_name))

                # Shuffle data
                ind = np.random.permutation((len(X)))
                X = X[ind]
                y = y[ind]

                # Normalize data!
                X = (X / 127.5) - 1.0
                # y to torch tensor:
                y = torch.from_numpy(y).float()

                batch_count = int(len(X) / batch_size)
                if len(X) / batch_size > batch_count:
                    batch_count += 1

                # So batchnorm doesnt receive a batch of 1
                if len(X) % batch_size == 1:
                    batch_count -= 1

                for j in range(batch_count):
                    self.optimizer.zero_grad()

                    batch_X = X[j * batch_size: min((j + 1) * batch_size, len(X))]
                    batch_y = y[j * batch_size: min((j + 1) * batch_size, len(X))].to(self.device)

                    predictions = self.predict(batch_X, eval=False).flatten()

                    prediction_loss = self.loss(predictions, batch_y)

                    current_average = (nr_frames / (
                                nr_frames + len(batch_X))) * current_average + prediction_loss.item() / (
                                                  nr_frames + len(batch_X))
                    nr_frames += len(batch_X)

                    prediction_loss.backward()

                    # torch.nn.utils.clip_grad_norm(self.brain.parameters(), 0.01)
                    self.optimizer.step()
                    self.lr_scheduler.step(epoch + j / batch_count)

                X, y = None, None  # De-allocate before validation and next load
                if verbose > 1:
                    print(f"{100 * (i + 1) / data_len}% epoch completion...")

            epoch_train_losses.append(current_average)

            if verbose > 0:
                print(f"Average loss for epoch #{epoch} : {current_average}")
                # Check validation loss
                if validation is True:
                    val_loss = self.validation(100, monitor_dynamics=monitor_val_predictions)
                    epoch_test_losses.append(val_loss)
                    print(f"Validation loss for epoch #{epoch} : {val_loss}")
                    # Put back into training mode
                    self.brain.train()

            # Save model:
            if save_mode == 'all':
                self.save_brain('prime_control', 'pc_optimizer_state')
            if save_mode == 'best' and val_loss < min_val_loss:
                min_val_loss = val_loss
                self.save_brain('best_prime_control', 'best_pc_optimizer_state')

        if graph_time > 0:
            graph_loss(epoch_train_losses, epoch_test_losses, graph_time)
