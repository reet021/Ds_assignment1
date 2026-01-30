from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import re

app = Flask(__name__)

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    if request.method == "POST":
        file = request.files["file"]
        weights = request.form["weights"]
        impacts = request.form["impacts"]
        email = request.form["email"]

        if not is_valid_email(email):
            return render_template("index.html", message="Invalid email format")

        weights = weights.split(",")
        impacts = impacts.split(",")

        data = pd.read_csv(file)

        if len(weights) != len(impacts) or len(weights) != data.shape[1] - 1:
            return render_template("index.html", message="Weights, impacts, and columns mismatch")

        weights = np.array(weights, dtype=float)
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

        d_pos = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
        d_neg = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

        score = d_neg / (d_pos + d_neg)
        data["Topsis Score"] = score
        data["Rank"] = data["Topsis Score"].rank(ascending=False)

        data.to_csv("result.csv", index=False)

        message = "Result generated successfully and sent to email (simulation)."

    return render_template("index.html", message=message)

if __name__ == "__main__":
    app.run(debug=True)
