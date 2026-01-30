import sys
import pandas as pd
import numpy as np
import os

def error(msg):
    print("Error:", msg)
    sys.exit(1)

def main():
    if len(sys.argv) != 5:
        error("Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <OutputFile>")

    input_file = sys.argv[1]
    weights = sys.argv[2].split(",")
    impacts = sys.argv[3].split(",")
    output_file = sys.argv[4]

    if not os.path.exists(input_file):
        error("Input file not found")

    data = pd.read_csv(input_file)

    if data.shape[1] < 3:
        error("Input file must contain at least 3 columns")

    weights = np.array(weights, dtype=float)

    for i in impacts:
        if i not in ['+', '-']:
            error("Impacts must be + or -")

    if len(weights) != len(impacts) or len(weights) != data.shape[1] - 1:
        error("Number of weights, impacts and columns must match")

    values = data.iloc[:, 1:].astype(float).values

    norm = values / np.sqrt((values ** 2).sum(axis=0))
    weighted = norm * weights

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

    d_pos = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    score = d_neg / (d_pos + d_neg)

    data['Topsis Score'] = score
    data['Rank'] = data['Topsis Score'].rank(ascending=False)

    data.to_csv(output_file, index=False)
    print("Result saved to", output_file)

if __name__ == "__main__":
    main()
