import torch
import torch.nn.functional as F
import torchvision as tv
import torch.nn as nn
import os

# Đảm bảo thư mục logs tồn tại
os.makedirs("logs", exist_ok=True)

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)
    def forward(self, x):
        return self.linear(x.view(-1, 28 * 28))

def evaluate_model(weights, round_num=None):
    # Tạo model và nạp trọng số
    model = LogisticRegression()
    state_dict = model.state_dict()
    for k, w in zip(state_dict.keys(), weights):
        state_dict[k] = torch.tensor(w)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Tải tập test MNIST
    test_dataset = tv.datasets.MNIST(root="./data", train=False, transform=tv.transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = total_loss / total
    print(f"Test accuracy: {accuracy:.4f}, Test loss: {avg_loss:.4f}")
    print(f"correct: {correct}, total: {total}, total_loss: {total_loss:.4f}, accuracy: {accuracy:.4f}")

    log_lines = []
    if round_num is not None:
        log_lines.append(f"Round {round_num}:")
    log_lines.append(f"  Test accuracy : {accuracy:.4f}")
    log_lines.append(f"  Test loss     : {avg_loss:.4f}")
    log_lines.append(f"  Correct       : {correct}")
    log_lines.append(f"  Total         : {total}")
    log_lines.append(f"  Total loss    : {total_loss:.4f}")
    log_lines.append("")  # Thêm dòng trống phân cách

    with open("../logs/test_result_log.txt", "a") as f:
        f.write("\n".join(log_lines))
    return accuracy, avg_loss