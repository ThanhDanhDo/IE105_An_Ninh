import flwr as fl
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

# Define a simple model (Logistic Regression)
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.linear(x.view(-1, 28 * 28))

# Load MNIST dataset
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    return trainloader

# Define Flower client
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader):
        self.model = model
        self.trainloader = trainloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # One epoch
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        # Ghi log weights plaintext
        log_path = os.path.join("..", "logs", "plaintext_logs.txt")
        with open(log_path, "a") as f:
            # Chuyển các weights thành list để dễ đọc log
            weights_list = [w.tolist() for w in self.get_parameters(config)]
            f.write(f"Client weights (plaintext): {weights_list}\n")
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.trainloader.dataset), {}

if __name__ == "__main__":
    model = LogisticRegression()
    trainloader = load_data()
    client = FLClient(model, trainloader)
    print("Connecting to server...")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    print("Client connected to server.")