import torch
from torch import nn
import cv2
from torchvision import models, transforms
import glob
from sklearn.model_selection import train_test_split

LEARNING_RATE = 0.001
EPOCHS = 50

class ResnetLstm(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(ResnetLstm, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.flatten = nn.Flatten()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, dropout=0.2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size*2),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size*2, out_features=hidden_size//2),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size//2, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, sequence: torch.Tensor):
        # sequence = sequence.squeeze(dim=1).to(torch.device("cuda"))
        sequence = self.flatten(self.conv(sequence))
        result, _ = self.lstm(sequence)
        output = self.head(result)
        return output

def accuracy_fn(y_pred, y):
      return sum([1 for i in range(len(y)) if (y_pred[i] < 0.5 and y[i] == 0) or (y_pred[i] >= 0.5 and y[i] == 1)]) / len(y)

def train(model: ResnetLstm, X: list, Y: list, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, epochs: int, accuracy_fn, device: torch.device="cpu"):
    model.to(device)
    resnet = nn.Sequential(*(list(models.resnet18(pretrained=True).children())[0:8])).to(device)
    resnet.eval()
    print("\n\n===============Beginning the Training Process==================")
    print(f"Size of training dataset: {len(X)} | No of Epochs: {epochs}")
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1} | ", end=" ")
        train_loss = 0
        accuracy = 0
        for (x, y) in zip(X, Y):
            video = cv2.VideoCapture(x)
            sequence = []
            with torch.no_grad():
              while video.isOpened():
                  ok, frame = video.read()
                  if not ok:
                      break
                  # frame = Image.fromarray(frame)
                  frame = transforms.ToTensor()(frame).to(device)
                  frame = resnet(frame.unsqueeze(dim=0))
                  sequence.append(frame.squeeze())
            sequence = torch.stack(sequence)
            sequence.to(device)
            y = torch.Tensor([y]*len(sequence))
            y = y.to(device)
            y_pred = model(sequence).squeeze()
            loss = loss_fn(y_pred, y)
            train_loss += float(loss)
            accuracy += accuracy_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Train Loss: {train_loss / len(X)} | Train Accuracy: {accuracy / len(X)}")

def test(model: ResnetLstm, X: list, Y: list, loss_fn: nn.Module, accuracy_fn, device: torch.device="cpu"):
    resnet = nn.Sequential(*(list(models.resnet18(pretrained=True).children())[0:8])).to(device)
    resnet.eval()
    print("\n===========Beginning the Testing Process====================")
    print(f"Size of test dataset: {len(X)}")
    model.to(device)
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.inference_mode():
        for (x, y) in zip(X, Y):
            video = cv2.VideoCapture(x)
            sequence = []
            while video.isOpened():
                ok, frame = video.read()
                if not ok:
                    break
                # frame = Image.fromarray(frame)
                frame = transforms.ToTensor()(frame).to(device)
                frame = resnet(frame.unsqueeze(dim=0))
                sequence.append(frame.squeeze())
            sequence = torch.stack(sequence)
            sequence.to(device)
            y = torch.Tensor([y]*len(sequence))
            y = y.to(device)
            y_pred = model(sequence).squeeze()
            loss = loss_fn(y_pred, y)
            test_loss += float(loss)
            accuracy += accuracy_fn(y_pred, y)
        print(f"Test Loss: {test_loss / len(X)} | Test Accuracy: {accuracy / len(X)}")

def get_data():
    fake = glob.glob("./dataset/fake/*")
    real = glob.glob("./dataset/real/*")
    y = [0]*len(fake) + [1]*len(real)
    X = fake + real
    print(f"Fake videos: {len(fake)} | Real Videos: {len(real)} | Total Videos: {len(y)}")
    return X, y


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device in use: {device}")
    model = ResnetLstm(input_size=420, hidden_size=1000)
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print(f"Total Trainable parameters is {total_params}")
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    train(model, X_train[0], y_train[0], loss_fn, optimizer, EPOCHS, accuracy_fn, device)
    test(model, X_test[0], y_test[0], loss_fn, accuracy_fn, device)

if __name__ == "__main__":
    main()