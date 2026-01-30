import sys
import pandas as pd
import numpy as np
import os

def error(msg):
    print("Error:", msg)
    sys.exit(1)

def main():
    # Check number of arguments
    if len(sys.argv) != 5:
        error("Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <OutputFile>")

    input_file = sys.argv[1]
    weights = sys.argv[2].split(",")
    impacts = sys.argv[3].split(",")
    output_file = sys.argv[4]

    # Check if file exists
    if not os.path.exists(input_file):
        error("Input file not found")

    # Read CSV
    try:
        data = pd.read_csv(input_file)
    except:
        error("Unable to read input file")

    # Check minimum columns
    if data.shape[1] < 3:
        error("Input file must contain at least 3 columns")

    # Convert weights to float
    try:
        weights = np.array(weights, dtype=float)
    except:
        error("Weights must be numeric and comma separated")

    # Validate impacts
    for i in impacts:
        if i not in ['+', '-']:
            error("Impacts must be either + or -")

    # Check length match
    if len(weights) != len(impacts) or len(weights) != data.shape[1] - 1:
        error("Number of weights, impacts and criteria must be same")

    # Extract numeric values
    try:
        values = data.iloc[:, 1:].astype(float).values
    except:
        error("All columns except first must be numeric")

    # Normalize
    norm = values / np.sqrt((values ** 2).sum(axis=0))

    # Weighted normalized matrix
    weighted = norm * weights

    # Ideal best and worst
    ideal_best = []
    ideal_worst = []

    for i in range(len(impacts)):
        if impacts[i] == '+':
            ideal_best.append(weighted[:, i].max())
            ideal_worst.append(weighted[:, i].min())
        else:
            ideal_best.append(weighted[:, i].min())
            ideal_worst.append(weighted[:, i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # Distance calculation
    d_pos = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    # TOPSIS score
    score = d_neg / (d_pos + d_neg)

    # Add results
    data['Topsis Score'] = score
    data['Rank'] = data['Topsis Score'].rank(ascending=False)

    # Save output
    data.to_csv(output_file, index=False)
    print("Result saved to", output_file)

if __name__ == "__main__":
    main()
