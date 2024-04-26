import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target
feature_names = dataset.feature_names

# Access the target variable (data labels)
target = dataset.target
target_names = dataset.target_names

# Print the name of the target variable
print("target_names")
print(target_names)
# Print the feature names
print("Feature Names:")
print(feature_names)

df = pd.DataFrame(X_full, columns=feature_names)

# Calculate statistics for each feature using the describe() function
statistics = df.describe()

# Display the statistics for each feature
print(statistics)










import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target
feature_names = dataset.feature_names

features = ["MedInc", "AveOccup"]
features_idx = [feature_names.index(feature) for feature in features]
X = X_full[:, features_idx]

means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
percentiles_25 = np.percentile(X, 25, axis=0)
percentiles_50 = np.percentile(X, 50, axis=0)
percentiles_75 = np.percentile(X, 75, axis=0)

maximums = np.max(X, axis=0)
minimums = np.min(X, axis=0)

# Display the statistics for each feature
for i, feature in enumerate(features):
    print("Feature:", feature)
    print("Mean:", means[i])
    print("Standard Deviation:", stds[i])
    print("25th Percentile:", percentiles_25[i])
    print("50th Percentile (Median):", percentiles_50[i])
    print("75th Percentile:", percentiles_75[i])
    print("Maximum:", maximums[i])
    print("Minimum:", minimums[i])
    print("-" * 40)
