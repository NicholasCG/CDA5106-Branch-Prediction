import torch
from tqdm import tqdm
from glob import glob
import os
from collections import defaultdict, deque
import numpy as np

class TracesDataset(torch.utils.data.Dataset):
    def __init__(self, traces_path, mimics_path):
        '''
        Args:
            traces_path (str): Path to the traces file.
            mimics_path (str): Path to the mimics file.
        '''

        self.traces_path = traces_path
        # self.mimics_path = mimics_path

        traces = glob(traces_path)
        mimics = glob(mimics_path)

        # Sort the files to ensure they match
        traces.sort()
        mimics.sort()

        num_trues = 0

        self.outcomes = []

        for trace_file, mimics_file in tqdm(zip(traces, mimics), desc="Loading traces and mimics", unit="file", total=len(traces)):

            # If pt files exists, skip
            if os.path.exists(f"traces_{trace_file.split('/')[-1].split('.')[0]}.pt"):
                print(f"Trace file {trace_file} already processed. Skipping.")
                continue

            with open(trace_file, 'r') as f:
                trace = f.readlines()
            with open(mimics_file, 'r') as f:
                mimic = f.readlines()

            # 32 bit PC, 16-bit local history register, 512-bit global history register, 48-8 bit GAs
            # 16-bit GAs means 16 prior branch addresses with 8 bits for each address
            # These will all be combined into a single 944 bit vector

            # PC [0-31], GHR [32 - 544], LHR [544 - 560], GAs [560 - 944]

            local_history = defaultdict(lambda: 0)
            global_history = 0
            gas = deque([0] * 48, maxlen=48)

            for trace_line, mimic_line in tqdm(zip(trace[:50], mimic[:50]), desc="Processing traces", unit="line", total=len(trace)):
                pc, outcome = trace_line.split()
                pc = int(pc, 16)
                outcome = 1 if outcome == 't' else 0

                num_trues += outcome

                _, mimic_outcome = mimic_line.split()
                mimic_outcome = 1 if mimic_outcome == 't' else 0

                pc_local = pc & 0xFFFF

                bit_list = []
                for i in reversed(range(32)):
                    bit_list.append((pc >> i) & 1)
                for i in reversed(range(512)):
                    bit_list.append((global_history >> i) & 1)
                for i in reversed(range(16)):
                    bit_list.append((local_history[pc_local] >> i) & 1)
                for g in gas:
                    for i in reversed(range(8)):
                        bit_list.append((g >> i) & 1)
                input_vector = torch.tensor(bit_list, dtype=torch.float32)

                # Update global history (cut off to 512 bits)
                global_history = ((global_history << 1) | outcome) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
                # Update local history
                local_history[pc_local] = ((local_history[pc_local] << 1) | outcome) & 0xFFFF

                # Update global address history
                gas.append(pc & 0xFF)
                # Update outcomes
                self.outcomes.append((input_vector, outcome, mimic_outcome))

            torch.save(self.outcomes, f"traces_{trace_file.split('/')[-1].split('.')[0]}.pt")
            self.outcomes = []
            print(f"Trace file {trace_file} processed and saved.")

    def __len__(self):
        return len(self.outcomes)
    
    def __getitem__(self, idx):
        # Return the input vector and the outcome
        return self.outcomes[idx][0], self.outcomes[idx][1], self.outcomes[idx][2]
    
    
if __name__ == "__main__":

    traces = "traces/preprocess/trace_*.txt"
    mimics = "traces/preprocess/gshare_*.txt"

    dataset = TracesDataset(traces, mimics)
    print("All traces processed and saved.")

    # originals
    traces = "traces/originals/orig_*.txt"
    mimics = "traces/originals/gshare_*.txt"
    dataset = TracesDataset(traces, mimics)
    print("All traces processed and saved.")