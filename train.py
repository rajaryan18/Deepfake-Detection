import cv2
from torchvision import models, transforms
import torch
from torch import nn

from models import ResnetLstm, Convolution3d
from preprocess import preprocess, get_data, get_current, split_test

LEARNING_RATE = 0.02
ENSEMBLE = 20

def accuracy_fn(y_pred: torch.Tensor, y: torch.Tensor):
    return (y_pred.round() == y).float().mean()

def train(model: ResnetLstm, X: list, Y: list, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn: any, device: torch.device="cpu") -> None:
    model.train()
    model.to(device)
    resnet = nn.Sequential(*(list(models.resnet18(pretrained=True).children())[0:9])).to(device)
    resnet.eval()
    print("\n\n===============Beginning the Training Process==================")
    print(f"Size of training dataset: {len(X)}")
    train_loss = 0
    accuracy = 0
    for idx, (x, y) in enumerate(zip(X, Y)):
        video = cv2.VideoCapture(x)
        sequence = []
        # cnt=0
        with torch.no_grad():
            while video.isOpened():
                ok, frame = video.read()
                if not ok:
                    break
                # cnt += 1
                # if cnt%3 != 0:
                #     continue
                frame = preprocess(frame)
                frame = transforms.ToTensor()(frame).to(device)
                frame = resnet(frame.unsqueeze(dim=0))
                sequence.append(frame.squeeze())
        sequence = torch.stack(sequence)
        sequence.to(device)
        y = torch.Tensor([y]*len(sequence))
        y = y.to(device)
        y_pred = model(sequence).squeeze()
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += float(loss)
        accuracy += accuracy_fn(y_pred, y)
        if idx%10 == 0:
            print(f"Video: {idx} | Train Loss: {train_loss / (idx+1)} | Train Accuracy: {accuracy / (idx+1)} | Show Y: {y[0]} | Show y_pred: {y_pred[0]}")
        del sequence
    del resnet
    print(f"Train Loss: {train_loss / len(X)} | Train Accuracy: {accuracy / len(X)}")

def test(model: ResnetLstm, X: list, Y: list, loss_fn: nn.Module, accuracy_fn: any, device: torch.device="cpu"):
    resnet = nn.Sequential(*(list(models.resnet18(pretrained=True).children())[0:9])).to(device)
    resnet.eval()
    print("\n===========Beginning the Testing Process====================")
    print(f"Size of test dataset: {len(X)}")
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.inference_mode():
        for idx, (x, y) in enumerate(zip(X, Y)):
            video = cv2.VideoCapture(x)
            sequence = []
            # cnt = 0
            while video.isOpened():
                ok, frame = video.read()
                if not ok:
                    break
                # if cnt%3 != 0:
                #     continue
                frame = preprocess(frame)
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
            if idx%10 == 0:
                print(f"Video: {idx} | Test Loss: {test_loss / (idx+1)} | Test Accuracy: {accuracy / (idx+1)} | Show Y: {y[0]} | Show y_pred: {y_pred[0]}")
            del sequence
        del resnet
        print(f"Test Loss: {test_loss / len(X)} | Test Accuracy: {accuracy / len(X)}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device in use: {device}")
    
    # Model definition
    model = ResnetLstm(input_size=512, hidden_size=1024)
    # model = Convolution3d(input_size=500, hidden_size=1000)

    model.to(device)
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print(f"Total Trainable parameters is {total_params}")
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
    
    X, y = get_data()
    X, X_test, y, y_test = split_test(X, y, 0.3)

    for ensemble in range(ENSEMBLE):
        print(f"Ensemble {ensemble+1}:")
        X_train, X_val, y_train, y_val = get_current(X, y)
        train(model, X_train, y_train, loss_fn, optimizer, accuracy_fn, device)
        test(model, X_val, y_val, loss_fn, accuracy_fn, device)
    test(model, X_test, y_test, loss_fn, accuracy_fn, device)
    torch.save(model.state_dict(), "./models/resnetlstm.pt")

if __name__ == "__main__":
    main()