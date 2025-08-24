import pickle
import socket
import numpy as np
import zlib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

# Load Data
CLIENT_ID = 2  # Change to 2 for Client 2
DATA_FILE = f"client{CLIENT_ID}_data.csv"

print(f"[Client {CLIENT_ID}] Loading Data...")
X = np.loadtxt(DATA_FILE, delimiter=",")
y = np.loadtxt("labels.csv", delimiter=",").astype(int)

print(f"[Client {CLIENT_ID}] Data Loaded: X shape = {X.shape}, y shape = {y.shape}")

# Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train ExtraTreesClassifier to Get Gini Index Order
extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)
extra_trees.fit(X_scaled, y)

# Sort Features by Gini Index
sorted_indices = np.argsort(extra_trees.feature_importances_)[::-1]
X_sorted = X_scaled[:, sorted_indices]

# Extract Feature Embeddings
X_transformed = extra_trees.apply(X_sorted)

print(f"[Client {CLIENT_ID}] Transformed Features Shape: {X_transformed.shape}")

# Connect to Server and Send Data
print(f"[Client {CLIENT_ID}] Connecting to Server...")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)  # Increase send buffer size

try:
    client_socket.connect(("localhost", 9999))
    print(f"[Client {CLIENT_ID}] Connected to Server.")
    
    # Serialize and Compress Data
    data = pickle.dumps((X_transformed, y, sorted_indices))
    compressed_data = zlib.compress(data)  # Compress the serialized data
    
    # Send the length of the compressed data first
    data_length = len(compressed_data)
    client_socket.sendall(data_length.to_bytes(4, 'big'))  # Send length as 4 bytes
    
    # Send compressed data in chunks
    CHUNK_SIZE = 4096
    for i in range(0, data_length, CHUNK_SIZE):
        client_socket.sendall(compressed_data[i:i+CHUNK_SIZE])

    print(f"[Client {CLIENT_ID}] Compressed feature embeddings sent to Server in Gini order.")

except Exception as e:
    print(f"[Client {CLIENT_ID}] Connection Error: {e}")

finally:
    client_socket.close()
