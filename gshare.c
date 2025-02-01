/**
 * Sara Belal
 * EEL4768 Project 2
 * TO COMPILE: gcc -o SIM proj02.c -lm
 * TO EXECUTE: ./SIM [arg1] [arg2] [arg3]
 * 
 * where
 * arg1 = M (2^M entries in predictor table), 
 * arg2 = N (bits in GBH),
 * and arg3 = input file,
 * 
 * as per the assignment instructions for command line input.
 * 
 * remember to include -lm at end when compiling due to the use of <math.h> :-(
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int m;
int n;
int size;

int correct;
int incorrect;
int gbh = 0;
int predictor_table[999999];

/**
 * this function takes in the address of a branch instruction and whether or not the
 * branch was actually taken, simulates a branch prediction, and then checks the
 * prediction against the actual result and keeps count of if the prediction was
 * correct or incorrect.
 * 
 * @param add: address of the branch instruction in memory, used to point to predictors
 * @param taken: actual outcome of whether or not the branch was taken, from trace file
 */
void predict(long long int add, char taken) {
    //printf("your mom\n");
    long long int add_shifted = add >> 2; // get rid of last 2 bits of address
    int chop = (1 << m) - 1;              // create additional variable to get next M bits of address

    // XOR M bits of shifted address with the GBH at the most significant bits,
    // and less significant bits filled in with zeroes
    long long int index = (add_shifted & chop) ^ (gbh << (m - n)); 
    
    // update table based on actual outcome in the trace file
    if(taken == 't') {
        if(predictor_table[index] > 1) { // case: predict weakly or strongly taken
            correct++;
            if(predictor_table[index] < 3) predictor_table[index]++;
        }
        else {                           // case: predict weakly or strongly not taken
            incorrect++;
            predictor_table[index]++;
        }

        // update gbh to be shifted to right with 1 at most significant bit
        if (n > 0) {
            gbh = gbh >> 1;
            gbh = gbh | (1 << (n - 1));
        }
    }
    else {
        if(predictor_table[index] > 1) { // case: predict weakly or strongly taken
            incorrect++;                 // predicted taken, is not taken
            predictor_table[index]--;    // demote prediction away from taken
        }
        else {                           // case: predict weakly or strongly not taken
            correct++;
            if(predictor_table[index] > 0) predictor_table[index]--;
        }

        // update gbh to be shifted to right with 0 at most significant bit
        if(n > 0) gbh = gbh >> 1;
    }
}

int main(int argc, char **argv) {
    char taken;
    long long int add;

    // get arguments to configure predictor table
    int arg_ind = 0;

    // increment index until file name argument
    while (argv[arg_ind][0] == '.')
        arg_ind++;
    
    m = atoi(argv[arg_ind]);
    n = atoi(argv[arg_ind + 1]);
    char *file_path = argv[arg_ind + 2];

    FILE *file = fopen(file_path, "r");

    if (!file) {
        printf("Error: Could not open the trace file.\n");
        return 1;
    }

    size = pow(2, m); // initialize the predictor table size to 2^M entries

    // create and initialize predictor table
    for(int i = 0; i < size; i++) predictor_table[i] = 2;

    while (!feof(file)) {
        // read op and address
        fscanf(file, " %llx %c", &add, &taken);

        // begin the simulation for each address read
        predict(add, taken);
    }

    fclose(file);

    float ratio = ((float) incorrect) / ((float) (correct + incorrect));

    // print statistics
    printf("%d %d %f\n", m, n, ratio);

    return 0;
}