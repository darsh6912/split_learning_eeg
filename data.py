import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("32000/balanced_dataset.csv")

# Assuming the last column is the target label
X = df.iloc[:, :-1].values  # All columns except the last
y = df.iloc[:, -1].values   # Last column is the label

# Convert categorical labels to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to integers

# Split features into two sets for clients
num_features = X.shape[1]
split_index = num_features // 2  # Divide features into two halves

X_client1 = X[:, :split_index]  # First half of features
X_client2 = X[:, split_index:]  # Second half of features

# Save feature subsets and encoded labels as CSV
np.savetxt("client1_data.csv", X_client1, delimiter=",")
np.savetxt("client2_data.csv", X_client2, delimiter=",")
np.savetxt("labels.csv", y_encoded, delimiter=",", fmt="%d")  # Save as integers

# Save label mapping for reference
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# **Convert CSVs to NPY for efficient storage**
print("Converting CSV files to .npy format for better performance...")

# Client 1 Data
X1 = np.loadtxt("client1_data.csv", delimiter=",", dtype=np.float32)
np.save("client1_data.npy", X1)

# Client 2 Data
X2 = np.loadtxt("client2_data.csv", delimiter=",", dtype=np.float32)
np.save("client2_data.npy", X2)

# Labels
y_np = np.loadtxt("labels.csv", delimiter=",", dtype=np.int32)
np.save("labels.npy", y_np)

print("Data saved in .npy format. Now use `np.memmap` in the client script.")
