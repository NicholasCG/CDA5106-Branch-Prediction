from argparse import ArgumentParser

class Smith:
    def __init__(self, b):
        self.b = b
        self.counter = 1 << (b - 1)
        self.thres = 1 << (b - 1)

    def predict(self, pc):
        if self.counter >= self.thres:
            return "t"
        else:
            return "n"
        
    def update(self, outcome):
        if outcome == "t":
            self.counter = min(self.counter + 1, (1 << self.b) - 1)
        else:
            self.counter = max(self.counter - 1, 0)

class Bimodal:
    def __init__(self, m):
        self.m = m
        self.table = [4] * (1 << m)
        self.index = 0

    def predict(self, pc):
        # get bits m + 1 to 2
        self.index = (pc >> 2) & ((1 << self.m) - 1)

        if self.table[self.index] >= 4:
            return "t"
        else:
            return "n"

    def update(self, outcome):
        if outcome == "t":
            self.table[self.index] = min(self.table[self.index] + 1, 7)
        else:
            self.table[self.index] = max(self.table[self.index] - 1, 0)

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

class Hybrid:
    def __init__(self, k, m1, n, m2):
        self.k = k
        self.gshare = Gshare(m1, n)
        self.bimodal = Bimodal(m2)
        self.chooser_table = [1] * (1 << k)
        self.idx = 0

        self.gshare_output = None
        self.bimodal_output = None

    def predict(self, pc):
        self.idx = (pc >> 2) & ((1 << self.k) - 1)

        self.gshare_output = self.gshare.predict(pc)
        self.bimodal_output = self.bimodal.predict(pc)

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
            # Also update the global history of the gshare predictor
            self.gshare.global_history = (self.gshare.global_history >> 1) | (int(outcome == "t") << (self.gshare.n - 1))

        # Update the chooser table
        if self.gshare_output == outcome and self.bimodal_output != outcome:
            self.chooser_table[self.idx] = min(self.chooser_table[self.idx] + 1, 3)
        elif self.gshare_output != outcome and self.bimodal_output == outcome:
            self.chooser_table[self.idx] = max(self.chooser_table[self.idx] - 1, 0)

if __name__ == "__main__":
    parser = ArgumentParser(description="Branch Prediction")
    parser.add_argument("predictor", type=str, help="Predictor type")
    parser.add_argument("arg1", type=str, help="Argument 1")
    parser.add_argument("arg2", type=str, nargs='?', default=None, help="Argument 2")
    parser.add_argument("arg3", type=str, nargs='?', default=None, help="Argument 3")
    parser.add_argument("arg4", type=str, nargs='?', default=None, help="Argument 4")
    parser.add_argument("arg5", type=str, nargs='?', default=None, help="Argument 5")

    args = parser.parse_args()
    trace_file = None

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
    
    with open(trace_file, "r") as f:
        for line in f:
            number_branches += 1
            pc, outcome = line.split()
            pc = int(pc, 16)

            prediction = predictor.predict(pc)
            predictor.update(outcome)
            if prediction != outcome:
                number_mispredictions += 1

    print("number of predictions: ", number_branches)
    print("number of mispredictions: ", number_mispredictions)
    print(f"misprediction rate: {(number_mispredictions / number_branches) * 100:.2f}%")

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
