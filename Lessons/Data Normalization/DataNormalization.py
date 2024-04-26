

{
"cell_type": "code",
"execution_count": 8,
"metadata": {},
"outputs": [
{
    "data": {

    "text/plain": [
    "<Figure size 1000x500 with 2 Axes>"
    ]
    },
    "metadata": {},
    "output_type": "display_data"
},
{
    "name": "stdout",
    "output_type": "stream",
    "text": [
    "         MedInc  AveOccup\n",
    "count 20640.000 20640.000\n",
    "mean      0.154     0.296\n",
    "std       0.872    12.183\n",
    "min      -1.392    -2.494\n",
    "25%      -0.446    -0.456\n",
    "50%       0.000     0.000\n",
    "75%       0.554     0.544\n",
    "max       5.260  1455.116\n"
    ]
}
],
"source": [
"import pandas as pd\n",
"from sklearn.datasets import fetch_california_housing\n",
"from sklearn.preprocessing import RobustScaler\n",
"\n",
"# Load the California housing dataset\n",
"dataset = fetch_california_housing()\n",
"X_full, y_full = dataset.data, dataset.target\n",
"feature_names = dataset.feature_names\n",
"\n",
"# Select the desired features\n",
"features = [\"MedInc\", \"AveOccup\"]\n",
"features_idx = [list(feature_names).index(feature) for feature in features]\n",
"X = X_full[:, features_idx]\n",
"df = pd.DataFrame(X_full, columns=feature_names)\n",
"# Initialize the RobustScaler\n",
"scaler = RobustScaler(quantile_range=(25, 75))\n",
"\n",
"\n",
"# Apply RobustScaler Scaling to the selected features\n",
"X_scaled = scaler.fit_transform(X)\n",
"\n",
"# Plot the data before and after normalization\n",
"plt.figure(figsize=(10, 5))\n",
"\n",
"# Plot the data before normalization\n",
"plt.subplot(1, 2, 1)\n",
"plt.scatter(df[\"MedInc\"], df[\"AveOccup\"])\n",
"plt.xlabel(\"Median Income\")\n",
"plt.ylabel(\"Average Occupancy\")\n",
"plt.title(\"Data Before Normalization\")\n",
"\n",
"# Plot the data after normalization\n",
"plt.subplot(1, 2, 2)\n",
"plt.scatter(X_scaled[:, 0], X_scaled[:, 1])\n",
"plt.xlabel(\"Normalized Median Income\")\n",
"plt.ylabel(\"Normalized Average Occupancy\")\n",
"plt.title(\"Data After RobustScaler\")\n",
"\n",
"plt.tight_layout()\n",
"plt.show()\n",
"\n",
"\n",
"df_scaled = pd.DataFrame(X_scaled, columns=features)\n",
"statistics = df_scaled.describe()\n",
"\n",
"pd.options.display.float_format = '{:.3f}'.format\n",
"\n",
"# Display the statistics for each feature\n",
"print(statistics)\n",
"\n",
"X_RobustScaler=X_scaled\n"
]
},
{
"cell_type": "code",
"execution_count": 9,
"metadata": {},
"outputs": [
{
    "data": {

    "text/plain": [
    "<Figure size 1000x500 with 1 Axes>"
    ]
    },
    "metadata": {},
    "output_type": "display_data"
}
],
"source": [
"# Plot the data before and after normalization\n",
"plt.figure(figsize=(10, 5))\n",
"\n",
"\n",
"# Plot the data after normalization\n",
"plt.subplot(1, 2, 2)\n",
"plt.scatter(X_RobustScaler[:, 0], X_RobustScaler[:, 1])\n",
"plt.xlabel(\"Normalized Median Income\")\n",
"plt.ylabel(\"Normalized Average Occupancy\")\n",
"plt.title(\"Data After RobustScaler\")\n",
"plt.ylim(-2, 4)  # Set the y-axis limits for zooming\n",
"plt.xlim(-2, 4)  # Set the x-axis limits for zooming\n",
"plt.tight_layout()\n",
"plt.show()"
]
},
{
"cell_type": "code",
"execution_count": 11,
"metadata": {},
"outputs": [
{
    "data": {

    "text/plain": [
    "<Figure size 1500x500 with 3 Axes>"
    ]
    },
    "metadata": {},
    "output_type": "display_data"
}
],
"source": [
"# Plot the data after different scaling methods\n",
"plt.figure(figsize=(15, 5))\n",
"\n",
"# Plot the data after Min-Max Scaling\n",
"plt.subplot(1, 3, 1)\n",
"plt.scatter(X_MinMaxScaler[:, 0], X_MinMaxScaler[:, 1])\n",
"plt.xlim(-5, 5)  # Set the x-axis limits for zooming\n",
"plt.ylim(-5, 5)  # Set the y-axis limits for zooming\n",
"plt.xlabel(\"Normalized Median Income\")\n",
"plt.ylabel(\"Normalized Average Occupancy\")\n",
"plt.title(\"Data After Min-Max Scaling\")\n",
"\n",
"# Plot the data after Standard Scaling\n",
"plt.subplot(1, 3, 2)\n",
"plt.scatter(X_StandardScaler[:, 0], X_StandardScaler[:, 1])\n",
"plt.xlim(-5, 5)  # Set the x-axis limits for zooming\n",
"plt.ylim(-5, 5)  # Set the y-axis limits for zooming\n",
"plt.xlabel(\"Standardized Median Income\")\n",
"plt.ylabel(\"Standardized Average Occupancy\")\n",
"plt.title(\"Data After Standard Scaling\")\n",
"\n",
"# Plot the data after Robust Scaling\n",
"plt.subplot(1, 3, 3)\n",
"plt.scatter(X_RobustScaler[:, 0], X_RobustScaler[:, 1])\n",
"plt.xlim(-5, 5)  # Set the x-axis limits for zooming\n",
"plt.ylim(-5, 5)  # Set the y-axis limits for zooming\n",
"plt.xlabel(\"Robustly Scaled Median Income\")\n",
"plt.ylabel(\"Robustly Scaled Average Occupancy\")\n",
"plt.title(\"Data After Robust Scaling\")\n",
"\n",
"plt.tight_layout()\n",
"plt.show()\n",
"\n",
"\n",
"\n",
"\n"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": []
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"QuantileTransformer (uniform output)"
]
},
{
"cell_type": "code",
"execution_count": 10,
"metadata": {},
"outputs": [
{
    "data": {

    "text/plain": [
    "<Figure size 1000x500 with 2 Axes>"
    ]
    },
    "metadata": {},
    "output_type": "display_data"
},
{
    "name": "stdout",
    "output_type": "stream",
    "text": [
    "         MedInc  AveOccup\n",
    "count 20640.000 20640.000\n",
    "mean      0.500     0.500\n",
    "std       0.289     0.289\n",
    "min       0.000     0.000\n",
    "25%       0.250     0.250\n",
    "50%       0.500     0.500\n",
    "75%       0.750     0.750\n",
    "max       1.000     1.000\n"
    ]
}
],
"source": [
"\n",
"import pandas as pd\n",
"from sklearn.datasets import fetch_california_housing\n",
"from sklearn.preprocessing import QuantileTransformer\n",
"\n",
"# Load the California housing dataset\n",
"dataset = fetch_california_housing()\n",
"X_full, y_full = dataset.data, dataset.target\n",
"feature_names = dataset.feature_names\n",
"\n",
"# Select the desired features\n",
"features = [\"MedInc\", \"AveOccup\"]\n",
"features_idx = [list(feature_names).index(feature) for feature in features]\n",
"X = X_full[:, features_idx]\n",
"df = pd.DataFrame(X_full, columns=feature_names)\n",
"# Initialize the Scaler\n",
"scaler =  QuantileTransformer(output_distribution=\"uniform\")\n",
"\n",
"# Apply Scaling to the selected features\n",
"X_scaled = scaler.fit_transform(X)\n",
"\n",
"# Plot the data before and after normalization\n",
"plt.figure(figsize=(10, 5))\n",
"\n",
"# Plot the data before normalization\n",
"plt.subplot(1, 2, 1)\n",
"plt.scatter(df[\"MedInc\"], df[\"AveOccup\"])\n",
"plt.xlabel(\"Median Income\")\n",
"plt.ylabel(\"Average Occupancy\")\n",
"plt.title(\"Data Before Normalization\")\n",
"\n",
"# Plot the data after normalization\n",
"plt.subplot(1, 2, 2)\n",
"plt.scatter(X_scaled[:, 0], X_scaled[:, 1])\n",
"plt.xlabel(\"Normalized Median Income\")\n",
"plt.ylabel(\"Normalized Average Occupancy\")\n",
"plt.title(\"Data After Quantile Transformer Scaling (Uniform)\")\n",
"\n",
"plt.tight_layout()\n",
"plt.show()\n",
"\n",
"df_scaled = pd.DataFrame(X_scaled, columns=features)\n",
"statistics = df_scaled.describe()\n",
"pd.options.display.float_format = '{:.3f}'.format\n",
"# Display the statistics for each feature\n",
"print(statistics)\n",
"\n",
"X_QuantileTransformer=X_scaled\n"
]
},
{
"cell_type": "code",
"execution_count": 11,
"metadata": {},
"outputs": [
{
    "data": {

    "text/plain": [
    "<Figure size 1000x500 with 2 Axes>"
    ]
    },
    "metadata": {},
    "output_type": "display_data"
},
{
    "name": "stdout",
    "output_type": "stream",
    "text": [
    "         MedInc  AveOccup\n",
    "count 20640.000 20640.000\n",
    "mean      0.004    -0.000\n",
    "std       1.024     0.999\n",
    "min      -5.199    -5.199\n",
    "25%      -0.675    -0.675\n",
    "50%      -0.000    -0.000\n",
    "75%       0.674     0.675\n",
    "max       5.199     5.199\n"
    ]
}
],
"source": [
"\n",
"\n",
"\n",
"import pandas as pd\n",
"from sklearn.datasets import fetch_california_housing\n",
"from sklearn.preprocessing import QuantileTransformer\n",
"\n",
"# Load the California housing dataset\n",
"dataset = fetch_california_housing()\n",
"X_full, y_full = dataset.data, dataset.target\n",
"feature_names = dataset.feature_names\n",
"\n",
"# Select the desired features\n",
"features = [\"MedInc\", \"AveOccup\"]\n",
"features_idx = [list(feature_names).index(feature) for feature in features]\n",
"X = X_full[:, features_idx]\n",
"df = pd.DataFrame(X_full, columns=feature_names)\n",
"# Initialize the Scaler\n",
"scaler =  QuantileTransformer(output_distribution=\"normal\")\n",
"\n",
"# Apply Scaling to the selected features\n",
"X_scaled = scaler.fit_transform(X)\n",
"\n",
"# Plot the data before and after normalization\n",
"plt.figure(figsize=(10, 5))\n",
"\n",
"# Plot the data before normalization\n",
"plt.subplot(1, 2, 1)\n",
"plt.scatter(df[\"MedInc\"], df[\"AveOccup\"])\n",
"plt.xlabel(\"Median Income\")\n",
"plt.ylabel(\"Average Occupancy\")\n",
"plt.title(\"Data Before Normalization\")\n",
"\n",
"# Plot the data after normalization\n",
"plt.subplot(1, 2, 2)\n",
"plt.scatter(X_scaled[:, 0], X_scaled[:, 1])\n",
"plt.xlabel(\"Normalized Median Income\")\n",
"plt.ylabel(\"Normalized Average Occupancy\")\n",
"plt.title(\"Data After Quantile Transformer Scaling (Normal)\")\n",
"\n",
"plt.tight_layout()\n",
"plt.show()\n",
"\n",
"df_scaled = pd.DataFrame(X_scaled, columns=features)\n",
"statistics = df_scaled.describe()\n",
"pd.options.display.float_format = '{:.3f}'.format\n",
"# Display the statistics for each feature\n",
"print(statistics)\n",
"\n",
"X_QuantileTransformer=X_scaled\n"
]
}
],
"metadata": {
"kernelspec": {
"display_name": "Python 3",
"language": "python",
"name": "python3"
},
"language_info": {
"codemirror_mode": {
"name": "ipython",
"version": 3
},
"file_extension": ".py",
"mimetype": "text/x-python",
"name": "python",
"nbconvert_exporter": "python",
"pygments_lexer": "ipython3",
"version": "3.7.7"
},
"orig_nbformat": 4
},
"nbformat": 4,
"nbformat_minor": 2