from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from collections import defaultdict, deque
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

class Gshare:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.table = [4] * (1 << m)
        self.global_history = 0
        self.idx = 0

    def predict(self, pc):
        # calculate index (NOTE FIX THE INDEXING HERE I KNOW *SOMETHING* IS WRONG)
        self.idx = (pc >> 2) & ((1 << self.m) - 1)
        self.idx = self.idx ^ self.global_history

        if self.table[self.idx] >= 4:
            return "t"
        else:
            return "n"

    def update(self, outcome):
        if outcome == "t":
            self.table[self.idx] = min(self.table[self.idx] + 1, 7)
        else:
            self.table[self.idx] = max(self.table[self.idx] - 1, 0)
        
        # update global history
        self.global_history = (self.global_history >> 1) | (int(outcome == "t") << (self.n - 1))

class DBNLike(nn.Module):
    """
    A multi-layer feed-forward network approximating the structure of the DBN from the paper.
    For an actual DBN, you would implement/stitch together RBMs and do pre-training.
    """
    def __init__(self, input_dim=944, hidden_dims=(630, 270, 630, 944), output_dim=1):
        super(DBNLike, self).__init__()
        # Four hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.fc_out = nn.Linear(hidden_dims[3], output_dim)

    def _get_flattened_size(self, input_dim):
        # Create a dummy input to calculate the flattened size dynamically
        dummy_input = torch.zeros(1, 1, input_dim)
        with torch.no_grad():
            output = self.layers[:-3](dummy_input)  # Exclude the linear layers
        return output.numel()

    def forward(self, x):
        # x shape: (N, 944) if passed in as a 2D [batch, features]
        # or (N, 1, 944) if you keep it in 3D and flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc_out(x))
        return x
    
class LeNetLike(nn.Module):
    '''
    LeNet inspired 1D CNN
    - Two convolutional layers + two fully connected layers
    '''
    def __init__(self, input_dim=944, output_dim=1):
        self.input_dim = input_dim
        super(LeNetLike, self).__init__()

        # Start with input_dim = 944

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=20, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=20, out_channels=50, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(11_500, 500),
            nn.ReLU(),
            nn.Linear(500, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (N, 944) if passed in as a 2D [batch, features]
        
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.layers(x)
        return x
    
class AlexNetLike(nn.Module):
    '''
    AlexNet inspired 1D CNN
    - Five convolutional layers + three fully connected layers
    '''
    def __init__(self, input_dim=944, output_dim=1):
        self.input_dim = input_dim
        super(AlexNetLike, self).__init__()

        # Start with input_dim = 944

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=256, out_channels=384, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6_912, 500),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(500, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x shape: (N, 944) if passed in as a 2D [batch, features]
        
        x = x.unsqueeze(1)
        x = self.layers(x)
        return x

    
class TracesDataset(torch.utils.data.Dataset):
    def __init__(self, traces_path, mimics_path):
        '''
        Args:
            traces_path (str): Path to the traces file.
            mimics_path (str): Path to the mimics file.
        '''

        self.traces_path = traces_path
        # self.mimics_path = mimics_path

        self.outcomes = []

        traces = glob(traces_path)
        mimics = glob(mimics_path)

        for trace_file in tqdm(traces, desc="Loading traces", unit="file", total=len(traces)):
            traces = torch.load(trace_file)
            self.outcomes.extend(traces)

        # Sort the files to ensure they match
        # traces.sort()
        # mimics.sort()

        # num_trues = 0

        # for trace_file, mimics_file in tqdm(zip(traces, mimics), desc="Loading traces and mimics", unit="file", total=len(traces)):
        #     print(f"Processing {trace_file} and {mimics_file}...")
        #     # If pt files exists, skip
        #     if os.path.exists(f"traces_{trace_file.split('/')[-1].split('.')[0]}.pt"):
        #         print(f"Trace file {trace_file} already processed. Skipping.")
        #         continue

        #     with open(trace_file, 'r') as f:
        #         trace = f.readlines()
        #     with open(mimics_file, 'r') as f:
        #         mimic = f.readlines()

        #     # 32 bit PC, 16-bit local history register, 512-bit global history register, 48-8 bit GAs
        #     # 16-bit GAs means 16 prior branch addresses with 8 bits for each address
        #     # These will all be combined into a single 944 bit vector

        #     # PC [0-31], GHR [32 - 544], LHR [544 - 560], GAs [560 - 944]

        #     local_history = defaultdict(lambda: 0)
        #     global_history = 0
        #     gas = deque([0] * 48, maxlen=48)

        #     print(len(trace), len(mimic))   

        #     for trace_line, mimic_line in tqdm(zip(trace, mimic), desc="Processing traces", unit="line", total=len(trace)):
        #         pc, outcome = trace_line.split()
        #         pc = int(pc, 16)
        #         outcome = 1 if outcome == 't' else 0

        #         num_trues += outcome

        #         _, mimic_outcome = mimic_line.split()
        #         mimic_outcome = 1 if mimic_outcome == 't' else 0

        #         pc_local = pc & 0xFFFF

        #         bit_list = []
        #         for i in reversed(range(32)):
        #             bit_list.append((pc >> i) & 1)
        #         for i in reversed(range(512)):
        #             bit_list.append((global_history >> i) & 1)
        #         for i in reversed(range(16)):
        #             bit_list.append((local_history[pc_local] >> i) & 1)
        #         for g in gas:
        #             for i in reversed(range(8)):
        #                 bit_list.append((g >> i) & 1)
        #         input_vector = torch.tensor(bit_list, dtype=torch.float32)

        #         # Update global history (cut off to 512 bits)
        #         global_history = ((global_history << 1) | outcome) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        #         # Update local history
        #         local_history[pc_local] = ((local_history[pc_local] << 1) | outcome) & 0xFFFF

        #         # Update global address history
        #         gas.append(pc & 0xFF)
        #         # Update outcomes
        #         self.outcomes.append((input_vector, outcome, mimic_outcome))

        #     torch.save(self.outcomes, f"traces_{trace_file.split('/')[-1].split('.')[0]}.pt")
        #     self.outcomes = []
        #     print(f"Trace file {trace_file} processed and saved.")

        # # Convert outcomes to a tensor
        # print(self.outcomes)

        # # Save the dataset to a file
        # torch.save(self.outcomes, "traces_dataset.pt")

    def __len__(self):
        return len(self.outcomes)
    
    def __getitem__(self, idx):
        # Return the input vector and the outcome
        return self.outcomes[idx][0], self.outcomes[idx][1], self.outcomes[idx][2]

if __name__ == "__main__":

    # base_dataset = TracesDataset("traces/originals/orig*.txt", "traces/originals/gshare*.txt")

                

    # print(f"Misprediction Rate for ({m}, {n}): {(mis_predictions / total_predictions) * 100:.2f}%")

    # if mis_predictions / total_predictions < best_misprediction_rate:
    #     best_misprediction_rate = mis_predictions / total_predictions
    #     best_combination = (m, n)
    #     print(f"Best combination so far: m={m}, n={n}, Misprediction rate: {best_misprediction_rate:.2f}")

    benchmark_traces = glob("traces/*/*.pt")

    model_dbn = DBNLike()
    model_lenet = LeNetLike()
    model_alexnet = AlexNetLike()

    model_dbn_mimic = DBNLike()
    model_lenet_mimic = LeNetLike()
    model_alexnet_mimic = AlexNetLike()

    model_dbn.load_state_dict(torch.load("new_model_DBNLike.pt"))
    model_lenet.load_state_dict(torch.load("new_model_LeNetLike.pt"))
    model_alexnet.load_state_dict(torch.load("new_model_AlexNetLike.pt"))

    model_dbn_mimic.load_state_dict(torch.load("new_model_mimic_DBNLike.pt"))
    model_lenet_mimic.load_state_dict(torch.load("new_model_mimic_LeNetLike.pt"))
    model_alexnet_mimic.load_state_dict(torch.load("new_model_mimic_AlexNetLike.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dbn.to(device)
    model_lenet.to(device)
    model_alexnet.to(device)
    model_dbn_mimic.to(device)
    model_lenet_mimic.to(device)
    model_alexnet_mimic.to(device)

    model_dbn.eval()
    model_lenet.eval()
    model_alexnet.eval()
    model_dbn_mimic.eval()
    model_lenet_mimic.eval()
    model_alexnet_mimic.eval()

    for trace in benchmark_traces:
        print(f"Benchmarking {trace}...")

        test_dataset = TracesDataset(trace, trace)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)
        
        dbn_correct = 0
        lenet_correct = 0
        alexnet_correct = 0
        dbn_mimic_correct = 0
        lenet_mimic_correct = 0
        alexnet_mimic_correct = 0

        base_correct = 0

        total_items = 0
        
        with torch.no_grad():
            for inputs, outcomes, mimic_outcomes in tqdm(test_loader, desc="Validation", unit="batch", total=len(test_loader)):
                inputs, outcomes, mimic_outcomes = inputs.to(device), outcomes.to(device), mimic_outcomes.to(device)

                # outputs = model(inputs)
                # loss = criterion(outputs.squeeze(), outcomes.float()) + mimic_criterion(outputs.squeeze(), mimic_outcomes.float())
                # valid_loss += loss.item()
                # valid_correct += (outputs.squeeze().round() == outcomes.float()).float().sum().item()
                # valid_base_correct += (outputs.squeeze().round() == mimic_outcomes.float()).float().sum().item()
                # total_items += outcomes.size(0)

                dbn_outputs = model_dbn(inputs)
                lenet_outputs = model_lenet(inputs)
                alexnet_outputs = model_alexnet(inputs)
                dbn_mimic_outputs = model_dbn_mimic(inputs)
                lenet_mimic_outputs = model_lenet_mimic(inputs)
                alexnet_mimic_outputs = model_alexnet_mimic(inputs)
                dbn_correct += (dbn_outputs.squeeze().round() == outcomes.float()).float().sum().item()
                lenet_correct += (lenet_outputs.squeeze().round() == outcomes.float()).float().sum().item()
                alexnet_correct += (alexnet_outputs.squeeze().round() == outcomes.float()).float().sum().item()
                dbn_mimic_correct += (dbn_mimic_outputs.squeeze().round() == outcomes.float()).float().sum().item()
                lenet_mimic_correct += (lenet_mimic_outputs.squeeze().round() == outcomes.float()).float().sum().item()
                alexnet_mimic_correct += (alexnet_mimic_outputs.squeeze().round() == outcomes.float()).float().sum().item()

                base_correct += (outcomes.float() == mimic_outcomes.float()).float().sum().item()

                total_items += outcomes.size(0)

        dbn_correct /= total_items
        lenet_correct /= total_items
        alexnet_correct /= total_items
        dbn_mimic_correct /= total_items
        lenet_mimic_correct /= total_items
        alexnet_mimic_correct /= total_items
        base_correct /= total_items

        print("GShare Misprediction Rate: ", (1 - base_correct) * 100)
        print("DBN Misprediction Rate: ", (1 - dbn_correct) * 100)
        print("LeNet Misprediction Rate: ", (1 - lenet_correct) * 100)
        print("AlexNet Misprediction Rate: ", (1 - alexnet_correct) * 100)

        print("DBN Misprediction Rate (Mimic): ", (1 - dbn_mimic_correct) * 100)
        print("LeNet Misprediction Rate (Mimic): ", (1 - lenet_mimic_correct) * 100)
        print("AlexNet Misprediction Rate (Mimic): ", (1 - alexnet_mimic_correct) * 100)


    exit(0)

    trace_path = "traces/train/traces*.pt"

    train_dataset = TracesDataset(trace_path, trace_path)
    # valid_dataset = TracesDataset("traces/valid/traces*.pt", "traces/valid/traces*.pt")
    # valid_dataset = TracesDataset("traces/valid/trace_*.txt", "traces/valid/gshare_trace_*.txt")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=512, shuffle=False)

    # # model = DBNLike()
    # model = LeNetLike()
    # # model = AlexNetLike()

    for mimic_on in [False, True]:
        for model in [DBNLike, LeNetLike, AlexNetLike]:
            model = model()
            print(f"Model: {model.__class__.__name__}, Mimic: {mimic_on}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            criterion = nn.BCELoss()
            mimic_criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
            num_epochs = 5

            best_accuracy = 0.0
            best_model = None

            train_losses = []
            valid_losses = []

            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0

                curr_loss = 0.0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", total=len(train_loader))
                for inputs, outcomes, mimic_outcomes in pbar:
                    inputs, outcomes, mimic_outcomes = inputs.to(device), outcomes.to(device), mimic_outcomes.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    # loss = criterion(outputs.squeeze(), outcomes.float())
                    loss = criterion(outputs.squeeze(), outcomes.float()) + mimic_criterion(outputs.squeeze(), mimic_outcomes.float())
                    loss.backward()

                    optimizer.step()
                    running_loss += loss.item()
                    curr_loss = loss.item()


                    accuracy = (outputs.squeeze().round() == outcomes.float()).float().mean()

                    pbar.set_postfix(loss=f'{curr_loss:.2E}', accuracy=accuracy.item())

                train_losses.append(running_loss / len(train_loader))


                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

                # # Validation phase
                # model.eval()
                # valid_loss = 0.0
                # valid_correct = 0
                # total_items = 0

                # valid_base_correct = 0

                # with torch.no_grad():
                #     for inputs, outcomes, mimic_outcomes in tqdm(valid_loader, desc="Validation", unit="batch", total=len(valid_loader)):
                #         inputs, outcomes, mimic_outcomes = inputs.to(device), outcomes.to(device), mimic_outcomes.to(device)

                #         outputs = model(inputs)
                #         loss = criterion(outputs.squeeze(), outcomes.float())
                        
                #         if mimic_on:
                #             loss += mimic_criterion(outputs.squeeze(), mimic_outcomes.float())

                #         valid_loss += loss.item()
                #         valid_correct += (outputs.squeeze().round() == outcomes.float()).float().sum().item()
                #         valid_base_correct += (outputs.squeeze().round() == mimic_outcomes.float()).float().sum().item()
                #         total_items += outcomes.size(0)

                # valid_correct /= total_items
                # valid_loss /= len(valid_loader)
                # valid_base_correct /= total_items

                # print(f"Validation Base Misprediction Rate: { (1 - valid_base_correct) * 100:.2f}%")

                # if valid_correct > best_accuracy:
                #     best_accuracy = valid_correct
                #     best_model = model.state_dict()
                    
                # valid_losses.append(valid_loss)
                # print(f"Validation Loss: {valid_loss}, Misprediction Rate: { (1 - valid_correct) * 100:.2f}%")
                    
                # scheduler.step()

            print(f"Best Misprediction Rate: { (1 - best_accuracy) * 100:.2f}%")
            # Save the best model
            # if best_model is not None:
            #     model.load_state_dict(best_model)
            #     print("Best model loaded.")
            #     # Save the model
            #     file_name = f"new_best_model_{model.__class__.__name__}.pt"

            #     if mimic_on:
            #         file_name = f"new_best_model_mimic_{model.__class__.__name__}.pt"

            #     torch.save(model.state_dict(), file_name)

            file_name = f"new_model_{model.__class__.__name__}.pt"
            if mimic_on:
                file_name = f"new_model_mimic_{model.__class__.__name__}.pt"
            torch.save(model.state_dict(), file_name)
            
            # Plot and save the loss
            plt.plot(train_losses, label=f'{model.__class__.__name__} Train Loss')
            # plt.plot(valid_losses, label=f'{model.__class__.__name__} Validation Loss')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Loss over iterations')
            plt.yscale('log')

            file_name = f'new_loss_{model.__class__.__name__}.png'

            if mimic_on:
                file_name = f'new_loss_mimic_{model.__class__.__name__}.png'

            plt.savefig(file_name)
        