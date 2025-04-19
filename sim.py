from argparse import ArgumentParser

# Smith predictor: A simple counter-based predictor.
# It uses a saturating counter to predict whether a branch will be taken ("t") or not taken ("n").
class Smith:
    def __init__(self, b):
        self.b = b  # Number of bits in the counter
        self.counter = 1 << (b - 1)  # Initialize counter to mid-point
        self.thres = 1 << (b - 1)  # Threshold for predicting "t"

    def predict(self, pc):
        # Predict "t" if counter >= threshold, otherwise predict "n"
        if self.counter >= self.thres:
            return "t"
        else:
            return "n"
        
    def update(self, outcome):
        # Update the counter based on the actual outcome
        if outcome == "t":
            self.counter = min(self.counter + 1, (1 << self.b) - 1)  # Increment counter
        else:
            self.counter = max(self.counter - 1, 0)  # Decrement counter

# Bimodal predictor: Uses a table of counters indexed by the program counter (PC).
# Each counter predicts "t" or "n" based on its value.
class Bimodal:
    def __init__(self, m):
        self.m = m  # Number of index bits
        self.table = [4] * (1 << m)  # Initialize table with mid-point values
        self.index = 0

    def predict(self, pc):
        # Calculate the index using bits m+1 to 2 of the PC
        self.index = (pc >> 2) & ((1 << self.m) - 1)

        # Predict "t" if counter >= 4, otherwise predict "n"
        if self.table[self.index] >= 4:
            return "t"
        else:
            return "n"

    def update(self, outcome):
        # Update the counter at the calculated index
        if outcome == "t":
            self.table[self.index] = min(self.table[self.index] + 1, 7)  # Increment counter
        else:
            self.table[self.index] = max(self.table[self.index] - 1, 0)  # Decrement counter

# Gshare predictor: Combines global history with the PC to index into a table of counters.
class Gshare:
    def __init__(self, m, n):
        self.m = m  # Number of index bits
        self.n = n  # Number of global history bits
        self.table = [4] * (1 << m)  # Initialize table with mid-point values
        self.global_history = 0  # Initialize global history
        self.idx = 0

    def predict(self, pc):
        # Calculate the index by XORing the PC and global history
        self.idx = (pc >> 2) & ((1 << self.m) - 1)
        self.idx = self.idx ^ self.global_history

        # Predict "t" if counter >= 4, otherwise predict "n"
        if self.table[self.idx] >= 4:
            return "t"
        else:
            return "n"

    def update(self, outcome):
        # Update the counter at the calculated index
        if outcome == "t":
            self.table[self.idx] = min(self.table[self.idx] + 1, 7)  # Increment counter
        else:
            self.table[self.idx] = max(self.table[self.idx] - 1, 0)  # Decrement counter
        
        # Update the global history
        self.global_history = (self.global_history >> 1) | (int(outcome == "t") << (self.n - 1))

# Hybrid predictor: Combines Gshare and Bimodal predictors using a chooser table.
class Hybrid:
    def __init__(self, k, m1, n, m2):
        self.k = k  # Number of chooser table index bits
        self.gshare = Gshare(m1, n)  # Gshare predictor
        self.bimodal = Bimodal(m2)  # Bimodal predictor
        self.chooser_table = [1] * (1 << k)  # Initialize chooser table
        self.idx = 0

        self.gshare_output = None
        self.bimodal_output = None

    def predict(self, pc):
        # Calculate the chooser table index
        self.idx = (pc >> 2) & ((1 << self.k) - 1)

        # Get predictions from Gshare and Bimodal
        self.gshare_output = self.gshare.predict(pc)
        self.bimodal_output = self.bimodal.predict(pc)

        # Use chooser table to select the predictor
        if self.chooser_table[self.idx] >= 2:
            return self.gshare_output
        else:
            return self.bimodal_output

    def update(self, outcome):
        # Update the predictor used
        if self.chooser_table[self.idx] >= 2:
            self.gshare.update(outcome)
        else:
            self.bimodal.update(outcome)
            # Also update the global history of the Gshare predictor
            self.gshare.global_history = (self.gshare.global_history >> 1) | (int(outcome == "t") << (self.gshare.n - 1))

        # Update the chooser table based on prediction accuracy
        if self.gshare_output == outcome and self.bimodal_output != outcome:
            self.chooser_table[self.idx] = min(self.chooser_table[self.idx] + 1, 3)
        elif self.gshare_output != outcome and self.bimodal_output == outcome:
            self.chooser_table[self.idx] = max(self.chooser_table[self.idx] - 1, 0)

if __name__ == "__main__":
    # Main code: Parses arguments, initializes the predictor, and processes the trace file.
    parser = ArgumentParser(description="Branch Prediction")
    parser.add_argument("predictor", type=str, help="Predictor type")
    parser.add_argument("arg1", type=str, help="Argument 1")
    parser.add_argument("arg2", type=str, nargs='?', default=None, help="Argument 2")
    parser.add_argument("arg3", type=str, nargs='?', default=None, help="Argument 3")
    parser.add_argument("arg4", type=str, nargs='?', default=None, help="Argument 4")
    parser.add_argument("arg5", type=str, nargs='?', default=None, help="Argument 5")

    args = parser.parse_args()
    trace_file = None

    # Initialize the appropriate predictor based on the input arguments
    if args.predictor == "smith":
        predictor = Smith(int(args.arg1))
        trace_file = args.arg2
    elif args.predictor == "gshare":
        predictor = Gshare(int(args.arg1), int(args.arg2))
        trace_file = args.arg3
    elif args.predictor == "bimodal":
        predictor = Bimodal(int(args.arg1))
        trace_file = args.arg2
    elif args.predictor == "hybrid":
        predictor = Hybrid(int(args.arg1), int(args.arg2), int(args.arg3), int(args.arg4))
        trace_file = args.arg5
    else:
        print("Invalid predictor type")
        exit(1)

    number_branches = 0
    number_mispredictions = 0
    print("COMMAND")
    print("./sim", end="")
    for arg in vars(args).values():
        if arg is not None:
            print(f" {arg}", end="")
    print("\nOUTPUT")
    # Process the trace file
    with open(trace_file, "r") as f:
        for line in f:
            number_branches += 1
            pc, outcome = line.split()
            pc = int(pc, 16)  # Convert PC from hexadecimal to integer

            prediction = predictor.predict(pc)  # Get prediction
            predictor.update(outcome)  # Update predictor
            if prediction != outcome:
                number_mispredictions += 1  # Count mispredictions

    # Print results
    print("number of predictions: ", number_branches)
    print("number of mispredictions: ", number_mispredictions)
    print(f"misprediction rate: {(number_mispredictions / number_branches) * 100:.2f}%")

    # Print final contents of the predictor
    if args.predictor == "smith":
        print(f"FINAL COUNTER CONTENT: {predictor.counter}")
    elif args.predictor == "bimodal":
        print("FINAL BIMODAL CONTENTS")
        for i, entry in enumerate(predictor.table):
            print(f"{i} {entry}")
    elif args.predictor == "gshare":
        print("FINAL GSHARE CONTENTS")
        for i, entry in enumerate(predictor.table):
            print(f"{i} {entry}")
    elif args.predictor == "hybrid":
        print("FINAL CHOOSER CONTENTS")
        for i, entry in enumerate(predictor.chooser_table):
            print(f"{i} {entry}")
        print("FINAL GSHARE CONTENTS")
        for i, entry in enumerate(predictor.gshare.table):
            print(f"{i} {entry}")
        print("FINAL BIMODAL CONTENTS")
        for i, entry in enumerate(predictor.bimodal.table):
            print(f"{i} {entry}")
