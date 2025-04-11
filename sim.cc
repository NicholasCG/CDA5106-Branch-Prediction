#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <iomanip>

class Smith {
private:
    int b;
public:
    int counter;
    int thres;

    Smith(int bVal) : b(bVal) {
        counter = 1 << (b - 1);
        thres = 1 << (b - 1);
    }

    std::string predict() {
        return (counter >= thres) ? "t" : "n";
    }

    void update(const std::string &outcome) {
        if (outcome == "t") {
            counter = std::min(counter + 1, (1 << b) - 1);
        } else {
            counter = std::max(counter - 1, 0);
        }
    }
};

class Bimodal {
private:
    int m;
public:
    std::vector<int> table;
    int index;

    Bimodal(int mVal) : m(mVal), index(0) {
        table.resize(1 << m, 4);
    }

    std::string predict(unsigned pc) {
        index = (pc >> 2) & ((1 << m) - 1);
        if (table[index] >= 4) return "t";
        return "n";
    }

    void update(const std::string &outcome) {
        if (outcome == "t") {
            table[index] = std::min(table[index] + 1, 7);
        } else {
            table[index] = std::max(table[index] - 1, 0);
        }
    }
};

class Gshare {
private:
    int m, n;

public:
    int getN() const { return n; }
public:
    std::vector<int> table;
    unsigned global_history;
    int idx;

    Gshare(int mVal, int nVal) : m(mVal), n(nVal), global_history(0), idx(0) {
        table.resize(1 << m, 4);
    }

    std::string predict(unsigned pc) {
        idx = ((pc >> 2) & ((1 << m) - 1)) ^ global_history;
        return (table[idx] >= 4) ? "t" : "n";
    }

    void update(const std::string &outcome) {
        if (outcome == "t") {
            table[idx] = std::min(table[idx] + 1, 7);
        } else {
            table[idx] = std::max(table[idx] - 1, 0);
        }
        global_history = (global_history >> 1) | ((outcome == "t") << (n - 1));
    }
};

class Hybrid {
private:
    int k;
public:
    Gshare gshare;
    Bimodal bimodal;
    std::vector<int> chooser_table;
    int idx;
    std::string gshare_output;
    std::string bimodal_output;

    Hybrid(int kVal, int m1, int n, int m2)
     : k(kVal), gshare(m1, n), bimodal(m2), idx(0) {
        chooser_table.resize(1 << k, 1);
    }

    std::string predict(unsigned pc) {
        idx = (pc >> 2) & ((1 << k) - 1);
        gshare_output = gshare.predict(pc);
        bimodal_output = bimodal.predict(pc);
        return (chooser_table[idx] >= 2) ? gshare_output : bimodal_output;
    }

    void update(const std::string &outcome) {
        if (chooser_table[idx] >= 2) {
            gshare.update(outcome);
        } else {
            bimodal.update(outcome);
            // Update global history register regardless of which predictor is chosen
            gshare.global_history =
               (gshare.global_history >> 1) |
               ((outcome == "t") << (gshare.getN() - 1));
        }
        if (gshare_output == outcome && bimodal_output != outcome) {
            chooser_table[idx] = std::min(chooser_table[idx] + 1, 3);
        } else if (gshare_output != outcome && bimodal_output == outcome) {
            chooser_table[idx] = std::max(chooser_table[idx] - 1, 0);
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Invalid arguments.\n";
        return 1;
    }

    std::string predictorType = argv[1];
    std::string arg1 = argv[2], arg2, arg3, arg4, arg5;
    if (argc > 3) arg2 = argv[3];
    if (argc > 4) arg3 = argv[4];
    if (argc > 5) arg4 = argv[5];
    if (argc > 6) arg5 = argv[6];

    Smith* smithPtr = nullptr;
    Bimodal* bimodalPtr = nullptr;
    Gshare* gsharePtr = nullptr;
    Hybrid* hybridPtr = nullptr;
    std::string trace_file;

    if (predictorType == "smith") {
        smithPtr = new Smith(std::stoi(arg1));
        trace_file = arg2;
    } else if (predictorType == "gshare") {
        gsharePtr = new Gshare(std::stoi(arg1), std::stoi(arg2));
        trace_file = arg3;
    } else if (predictorType == "bimodal") {
        bimodalPtr = new Bimodal(std::stoi(arg1));
        trace_file = arg2;
    } else if (predictorType == "hybrid") {
        hybridPtr = new Hybrid(std::stoi(arg1),
                               std::stoi(arg2),
                               std::stoi(arg3),
                               std::stoi(arg4));
        trace_file = arg5;
    } else {
        std::cout << "Invalid predictor type\n";
        return 1;
    }

    std::ifstream fin(trace_file);
    if (!fin.is_open()) {
        std::cerr << "Could not open trace file.\n";
        return 1;
    }

    // Print command line information
    std::cout << "COMMAND\n";
    std::cout << "./sim";
    for (int i = 1; i < argc; i++) {
        std::cout << " " << argv[i];
    }
    std::cout << "\n";
    std::cout << "OUTPUT\n";

    int number_branches = 0;
    int number_mispredictions = 0;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        number_branches++;
        std::stringstream ss(line);
        std::string pcHex, outcome;
        ss >> pcHex >> outcome;
        unsigned pcVal = std::stoul(pcHex, nullptr, 16);

        std::string prediction;
        if (smithPtr) {
            prediction = smithPtr->predict();
            smithPtr->update(outcome);
        } else if (bimodalPtr) {
            prediction = bimodalPtr->predict(pcVal);
            bimodalPtr->update(outcome);
        } else if (gsharePtr) {
            prediction = gsharePtr->predict(pcVal);
            gsharePtr->update(outcome);
        } else if (hybridPtr) {
            prediction = hybridPtr->predict(pcVal);
            hybridPtr->update(outcome);
        }
        if (prediction != outcome) {
            number_mispredictions++;
        }
    }
    fin.close();

    std::cout << "number of predictions: " << number_branches << "\n";
    std::cout << "number of mispredictions: " << number_mispredictions << "\n";
    double rate = (number_branches == 0 ? 0.0
                  : 100.0 * number_mispredictions / number_branches);
    std::cout << "misprediction rate: " << std::fixed << std::setprecision(2)
              << rate << "%\n";

    if (smithPtr) {
        std::cout << "FINAL COUNTER CONTENT: " << smithPtr->counter << "\n";
    } else if (bimodalPtr) {
        std::cout << "FINAL BIMODAL CONTENTS\n";
        for (size_t i = 0; i < bimodalPtr->table.size(); i++) {
            std::cout << i << " " << bimodalPtr->table[i] << "\n";
        }
    } else if (gsharePtr) {
        std::cout << "FINAL GSHARE CONTENTS\n";
        for (size_t i = 0; i < gsharePtr->table.size(); i++) {
            std::cout << i << " " << gsharePtr->table[i] << "\n";
        }
    } else if (hybridPtr) {
        std::cout << "FINAL CHOOSER CONTENTS\n";
        for (size_t i = 0; i < hybridPtr->chooser_table.size(); i++) {
            std::cout << i << " " << hybridPtr->chooser_table[i] << "\n";
        }
        std::cout << "FINAL GSHARE CONTENTS\n";
        for (size_t i = 0; i < hybridPtr->gshare.table.size(); i++) {
            std::cout << i << " " << hybridPtr->gshare.table[i] << "\n";
        }
        std::cout << "FINAL BIMODAL CONTENTS\n";
        for (size_t i = 0; i < hybridPtr->bimodal.table.size(); i++) {
            std::cout << i << " " << hybridPtr->bimodal.table[i] << "\n";
        }
    }

    delete smithPtr;
    delete bimodalPtr;
    delete gsharePtr;
    delete hybridPtr;
    return 0;
}