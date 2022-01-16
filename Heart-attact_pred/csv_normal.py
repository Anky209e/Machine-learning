import numpy as np
import pandas as pd
import math

inputs = pd.read_csv("heart.csv")
# inputs.pop("Id")
targets = inputs.pop("target")

inputs = np.array(inputs)
inputs = inputs.transpose()

mean = []
std = []

for i in inputs:
    mean.append(i.mean())
    std.append(i.std())

print(mean)
print(std)