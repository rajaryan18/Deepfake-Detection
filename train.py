import glob
from sklearn.model_selection import train_test_split
import cv2
from torchvision import models, transforms
import torch
from torch import nn

from models import ResnetLstm, Convolution3d

LEARNING_RATE = 0.001
EPOCHS = 50

def accuracy_fn(y_pred: torch.Tensor, y: torch.Tensor) -> float:
      return sum([1 for i in range(len(y)) if (y_pred[i] < 0.5 and y[i] == 0) or (y_pred[i] >= 0.5 and y[i] == 1)]) / len(y)

def train(model: ResnetLstm, X: list, Y: list, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn: any, device: torch.device="cpu") -> None:
    model.to(device)
    resnet = nn.Sequential(*(list(models.resnet18(pretrained=True).children())[0:8])).to(device)
    resnet.eval()
    print("\n\n===============Beginning the Training Process==================")
    print(f"Size of training dataset: {len(X)}")
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

def test(model: ResnetLstm, X: list, Y: list, loss_fn: nn.Module, accuracy_fn: any, device: torch.device="cpu") -> None:
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

def get_data() -> tuple:
    fake = glob.glob("./dataset/fake/*")
    real = glob.glob("./dataset/real/*")
    y = [0]*len(fake) + [1]*len(real)
    X = fake + real
    print(f"Fake videos: {len(fake)} | Real Videos: {len(real)} | Total Videos: {len(y)}")
    return X, y


def main() -> None:
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
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}:")
        train(model, X_train[0], y_train[0], loss_fn, optimizer, accuracy_fn, device)
        test(model, X_test[0], y_test[0], loss_fn, accuracy_fn, device)

if __name__ == "__main__":
    main()