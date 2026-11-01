import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from time import time

def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    BATCH, EPOCHS = 64, 10

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(
            train, batch_size=BATCH, shuffle=True,
            num_workers=4, persistent_workers=True, drop_last=True
            )
    test_loader = DataLoader(test, batch_size=BATCH, num_workers=0)

    net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
            )

    model = torch.compile(net, backend="inductor", dynamic=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(EPOCHS):
        start = time()

        net.train()
        loss_sum = 0.

        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        end = time()

        print(f"Epoch {epoch+1}: loss = {loss_sum / len(train_loader):.4f}. Time it took: {end-start}s")

    net.eval()
    correct = total = 0

    with torch.inference_mode():
        for x, y in test_loader:
            correct += (net(x).argmax(1) == y).sum().item()
            total += y.size(0)

    print(f"Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
