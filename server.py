import pickle
import socket
import numpy as np
import zlib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("[Server] Starting Split Learning Server...")

# Set up Server Socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)  # Increase receive buffer size
server_socket.bind(("localhost", 9999))
server_socket.listen(2)  # Expecting 2 Clients

received_embeddings = []
sorted_indices_list = []
labels = None

# Receive Data from Clients
for i in range(2):  # Wait for 2 clients
    print(f"[Server] Waiting for client {i + 1}...")
    client_socket, addr = server_socket.accept()
    print(f"[Server] Connection established with {addr}")

    try:
        # First, receive the length of the incoming compressed data
        data_length = int.from_bytes(client_socket.recv(4), 'big')  # Read 4 bytes for length
        print(f"[Server] Expecting {data_length} bytes from Client {i + 1}")

        # Now, receive the actual data in chunks
        data = b""
        CHUNK_SIZE = 4096
        while len(data) < data_length:
            packet = client_socket.recv(min(CHUNK_SIZE, data_length - len(data)))
            if not packet:
                break
            data += packet

        if len(data) != data_length:
            raise ValueError(f"[Server] Incomplete data received from Client {i + 1}")

        # Decompress the received data
        decompressed_data = zlib.decompress(data)
        
        # Deserialize the data
        embeddings, y_part, sorted_indices = pickle.loads(decompressed_data)

        received_embeddings.append(embeddings)
        sorted_indices_list.append(sorted_indices)

        if labels is None:
            labels = y_part  # Store labels from the first client

        print(f"[Server] Received embeddings from Client {i + 1}")

    except (pickle.UnpicklingError, zlib.error, ValueError) as e:
        print(f"[Server] Error: {e}")

    client_socket.close()

# Validate Received Data
if len(received_embeddings) == 0:
    print("[Server] Error: No embeddings received. Exiting...")
    exit()

print("[Server] Merging feature embeddings...")
X_combined = np.hstack(received_embeddings)
print(f"[Server] Combined Embeddings Shape: {X_combined.shape}, Labels Shape: {labels.shape}")

# Split Dataset (70% Train, 15% Validation, 15% Test)
print("[Server] Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(X_combined, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"[Server] Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# Train Final Model on All Features
print("[Server] Training ExtraTreesClassifier on received embeddings...")
final_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)

# Evaluate Model
y_pred_train = final_model.predict(X_train)
y_pred_val = final_model.predict(X_val)
y_pred_test = final_model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
val_acc = accuracy_score(y_val, y_pred_val)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"[Server] Training Accuracy: {train_acc:.4f}")
print(f"[Server] Validation Accuracy: {val_acc:.4f}")
print(f"[Server] Test Accuracy: {test_acc:.4f}")

print("[Server] Training complete!")
server_socket.close()
