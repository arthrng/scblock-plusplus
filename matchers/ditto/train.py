"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from model import DittoModel
from dataset import DittoDataset


def training_step(model, device, train_loader, optimizer, criterion, alpha):
    """
    Perform a single training step on the model.

    :param model: The model to be trained.
    :param device: The device to run the training on (CPU or GPU).
    :param train_loader: DataLoader for the training dataset.
    :param optimizer: Optimizer for updating the model parameters.
    :param criterion: Loss function.
    :param alpha: Alpha parameter for the model.
    :return: The trained model.
    """
    model.train()
    running_loss = 0

    for i, batch in enumerate(train_loader):
        x1, x2, y = batch
        outputs = model(input_ids=x1.to(device), input_ids_aug=x2.to(device))
        loss = criterion(outputs.view(-1, 1), y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(y)

    train_loss = running_loss / len(train_loader.dataset)
    print(f"Training Loss: {train_loss}")

    return model


def validation_step(model, device, val_loader, criterion):
    """
    Perform a single validation step on the model.

    :param model: The model to be validated.
    :param device: The device to run the validation on (CPU or GPU).
    :param val_loader: DataLoader for the validation dataset.
    :param criterion: Loss function.
    :return: The validated model.
    """
    model.eval()
    running_loss = 0
    num_correct = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x1, x2, y = batch
            outputs = model(input_ids=x1.to(device), input_ids_aug=x2.to(device))
            y = y.to(device)
            outputs = outputs.to(device)
            loss = criterion(outputs.view(-1), y.float())

            num_correct += list((outputs.view(-1, 1) > 0.5) == y.view(-1, 1))
            all_labels += list(y.view(-1, 1))

            running_loss += loss.item() * len(y)

    val_loss = running_loss / len(val_loader.dataset)
    num_correct = [int(t.cpu()) for t in num_correct]
    all_labels = [int(t.cpu()) for t in all_labels]

    print(f"Validation Loss: {val_loss}")
    print(f"F1 Score: {f1_score(all_labels, num_correct)}")

    return model


def train(dataset_name, alpha):
    """
    Train the model for a specific dataset and alpha value.

    :param dataset_name: Name of the dataset.
    :param alpha: Alpha parameter for the model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = DittoDataset(path=f'./data/processed/{dataset_name}-train-pairs.json.gz', dataset=dataset_name)
    val_dataset = DittoDataset(path=f'./data/processed/{dataset_name}-val-pairs.json.gz', dataset=dataset_name)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=train_dataset.get_sampler(), collate_fn=train_dataset.pad)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, sampler=val_dataset.get_sampler(), collate_fn=val_dataset.pad)

    # Initialize model
    model = DittoModel(len_tokenizer=len(train_dataset.tokenizer), alpha=alpha)
    model.load_state_dict(torch.load(f'./matchers/ditto/saved_matchers/{dataset_name}/ditto-{alpha}.pt', map_location=device), strict=False)
    model = model.to(device)

    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = nn.BCELoss().to(device)

    # Train and validate the model for a single epoch
    for epoch in range(1):
        print(f"EPOCH: {epoch}")
        model = training_step(model=model, device=device, train_loader=train_loader, optimizer=optimizer, criterion=criterion, alpha=alpha)
        validation_step(model=model, device=device, val_loader=val_loader, criterion=criterion)

    # Save the trained model
    torch.save(model.state_dict(), f'./matchers/ditto/saved_matchers/{dataset_name}/ditto-{alpha}.pt')


if __name__ == '__main__':
    dataset_names = ['amazon-google']  # Add more datasets if needed
    alphas = list(np.round(np.arange(0.1, 1.0, 0.1), 2))

    for dataset_name in dataset_names:
        print(f"Training on dataset: {dataset_name}")
        for alpha in alphas:
            print(f"Alpha: {alpha}")
            train(dataset_name, alpha=alpha)
