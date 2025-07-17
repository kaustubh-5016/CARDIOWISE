import argparse
import time
import warnings
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Check for CUDA device, default to CPU if not available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 1. Model Definition (LSTM) ---
# As specified in the project proposal, an LSTM is used for time-series analysis.
class LSTMNet(nn.Module):
    """A simple LSTM model for cardiac risk classification."""
    def __init__(self, input_dim=3, hidden_dim=32, n_layers=2, output_dim=1):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(DEVICE)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start of the dataset
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        
        return out

# --- 2. Data Generation and Loading ---
def get_simulated_sensor_data(num_samples=1000, sequence_length=60, client_id=1):
    """
    Simulates physiological data from ECG, BP, and SpO2 sensors.
    In a real implementation, this function would be replaced with code that reads
    data from the actual sensors connected to the Raspberry Pi's GPIO pins.
    
    Args:
        num_samples (int): The number of data points to generate.
        sequence_length (int): The length of each time-series sequence.
        client_id (int): A unique ID for the client to introduce data heterogeneity.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing features (X) and labels (y).
    """
    print(f"Generating simulated data for client {client_id}...")
    
    # Base data generation
    time_steps = np.linspace(0, 10 * np.pi, sequence_length)
    X =
    y =

    for _ in range(num_samples):
        # Simulate ECG signal (sinusoidal with some noise)
        ecg_signal = np.sin(time_steps + np.random.uniform(-0.5, 0.5)) + np.random.normal(0, 0.1, sequence_length)
        
        # Simulate Blood Pressure (stable with slight variations)
        bp_signal = np.full(sequence_length, 120 + np.random.normal(0, 5) - (client_id * 2)) # Introduce heterogeneity
        
        # Simulate SpO2 (stable with occasional dips for risky profiles)
        spo2_signal = np.full(sequence_length, 98 + np.random.normal(0, 0.5))
        
        # Create a label based on a simple rule (e.g., low SpO2 dip)
        is_risky = np.random.rand() > 0.8 # 20% chance of being a risky sample
        if is_risky:
            dip_start = np.random.randint(0, sequence_length - 10)
            spo2_signal[dip_start:dip_start+10] -= 5 # Simulate a dip
            label = 1
        else:
            label = 0
            
        # Combine features
        features = np.stack([ecg_signal, bp_signal, spo2_signal], axis=1)
        X.append(features)
        y.append(label)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

def load_data(client_id):
    """Loads the dataset for a given client."""
    X, y = get_simulated_sensor_data(client_id=client_id)
    
    # Split data into training and validation sets (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    return train_loader, val_loader, len(train_dataset)

# --- 3. Model Training and Evaluation Functions ---
def train(net, trainloader, epochs):
    """Train the neural network on the training set."""
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for sequences, labels in tqdm(trainloader, "Training"):
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the neural network on the entire test set."""
    criterion = torch.nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for sequences, labels in tqdm(testloader, "Testing"):
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            outputs = net(sequences)
            loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss / len(testloader.dataset), accuracy

# --- 4. Flower Client Implementation ---
class CardioWiseClient(fl.client.NumPyClient):
    """Flower client for the CARDIOWISE project."""
    def __init__(self, net, trainloader, valloader, num_examples):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_examples = num_examples

    def get_parameters(self, config):
        """Return the model's current parameters."""
        print("Client: Sending model parameters to the server.")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Update the local model with parameters from the server."""
        print("Client: Receiving model parameters from the server.")
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Train the local model using the parameters from the server,
        and return the updated local model parameters.
        """
        print("\n--- Client Training Round ---")
        self.set_parameters(parameters)
        
        # Train the model
        train(self.net, self.trainloader, epochs=1)
        
        # Return the updated parameters and metadata
        return self.get_parameters(config={}), self.num_examples, {}

    def evaluate(self, parameters, config):
        """
        Evaluate the local model using the parameters from the server.
        """
        print("\n--- Client Evaluation Round ---")
        self.set_parameters(parameters)
        
        # Evaluate the model
        loss, accuracy = test(self.net, self.valloader)
        
        print(f"Client Evaluation: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        return float(loss), self.num_examples, {"accuracy": float(accuracy)}

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARDIOWISE Flower Client")
    parser.add_argument(
        "--server_address",
        type=str,
        default="127.0.0.1:8080",
        help="Address of the federated learning server (default: 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--client_id",
        type=int,
        required=True,
        help="A unique integer ID for this client.",
    )
    args = parser.parse_args()

    print(f"Starting client {args.client_id}...")
    time.sleep(3) # Give server time to start

    # Load model and data
    net = LSTMNet().to(DEVICE)
    trainloader, valloader, num_examples = load_data(client_id=args.client_id)

    # Create the Flower client
    client = CardioWiseClient(net, trainloader, valloader, num_examples)

    # Start the Flower client
    # This will connect to the server, download the global model, train locally,
    # and upload the new weights.
    print("Connecting to the federated learning server...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )
    print("Client disconnected.")
