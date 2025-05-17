import flwr as fl
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from aes_utils import encrypt_bytes, decrypt_bytes, get_key_iv

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.linear(x.view(-1, 28 * 28))

def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    return trainloader

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader):
        self.model = model
        self.trainloader = trainloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        key, iv = get_key_iv()
        # Lấy shapes từ model gốc
        shapes = [v.shape for v in self.model.state_dict().values()]
        # Nếu là float32 (vòng đầu), không giải mã
        if parameters[0].dtype == np.float32:
            decrypted = parameters
        else:
            # Nếu là uint8 (ciphertext), giải mã và reshape lại đúng shape
            decrypted = [
                np.frombuffer(
                    decrypt_bytes(w.tobytes(), key, iv), dtype=np.float32
                ).reshape(shape)
                for w, shape in zip(parameters, shapes)
            ]
        params_dict = zip(self.model.state_dict().keys(), decrypted)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        weights = self.get_parameters(config)
        key, iv = get_key_iv()
        # Mã hóa và chuyển thành numpy array uint8
        encrypted_weights = [np.frombuffer(encrypt_bytes(w.tobytes(), key, iv), dtype=np.uint8) for w in weights]
        # Ghi log ciphertext
        log_path = os.path.join("..", "logs", "ciphertext_logs.txt")
        with open(log_path, "a") as f:
            f.write(f"Client weights (ciphertext): {[w.tobytes().hex() for w in encrypted_weights]}\n")
        return encrypted_weights, len(self.trainloader.dataset), {}

if __name__ == "__main__":
    model = LogisticRegression()
    trainloader = load_data()
    client = FLClient(model, trainloader)
    print("Connecting to server...")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    print("Client connected to server.")