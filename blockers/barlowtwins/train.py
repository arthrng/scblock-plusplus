import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import BarlowTwinsPretrainModel
from dataset import BarlowTwinsPretrainDataset

def off_diagonal(x):
    """
    Return a flattened view of the off-diagonal elements of a square matrix.
    
    Args:
        x (torch.Tensor): A square matrix of shape (n, n).

    Returns:
        torch.Tensor: A flattened tensor containing the off-diagonal elements.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    """
    Barlow Twins model for self-supervised learning.
    
    Reference:
        Barlow Twins: https://arxiv.org/abs/2103.03230
        Source code: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, lambd, model_dim):
        """
        Initialize the Barlow Twins model.

        Args:
            lambd (float): The lambda parameter to weigh the off-diagonal loss term.
            model_dim (int): Dimensionality of the model's input features.
        """
        super().__init__()
        self.lambd = lambd
        self.model_dim = model_dim

        # Define the projector network
        sizes = [self.model_dim, 4096]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-1], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # Normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, features):
        """
        Forward pass through the Barlow Twins model.

        Args:
            features (torch.Tensor): Input features of shape (batch_size, 2, model_dim).

        Returns:
            torch.Tensor: The computed loss.
        """
        z1 = self.projector(features[:, 0, :])
        z2 = self.projector(features[:, 1, :])

        # Empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() * (1.0 / 256)
        off_diag = off_diagonal(c).pow_(2).sum() * (1.0 / 256)
        loss = on_diag + self.lambd * off_diag

        return loss

def training_step(model, device, train_loader, optimizer, criterion):
    """
    Perform a single training step.

    Args:
        model (nn.Module): The model to be trained.
        device (torch.device): The device to run the model on.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model parameters.
        criterion (nn.Module): Loss function.

    Returns:
        nn.Module: The updated model after the training step.
    """
    model.train()
    running_loss = 0
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', additional_special_tokens=('[COL]', '[VAL]'))

    for (batch1, batch2) in train_loader:
        # Tokenize the input batches
        batch1 = tokenizer(list(batch1), truncation=True, padding=True)
        batch2 = tokenizer(list(batch2), truncation=True, padding=True)

        input_ids_left = torch.tensor(batch1['input_ids']).to(device)
        attention_mask_left = torch.tensor(batch1['attention_mask']).to(device)
        input_ids_right = torch.tensor(batch2['input_ids']).to(device)
        attention_mask_right = torch.tensor(batch2['attention_mask']).to(device)

        # Forward pass through the model
        outputs = model(input_ids_left=input_ids_left,
                        attention_mask_left=attention_mask_left,
                        input_ids_right=input_ids_right,
                        attention_mask_right=attention_mask_right)

        # Compute the loss
        loss = criterion(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(batch1)
    
    train_loss = running_loss / len(train_loader.dataset)
    print(train_loss)
    
    return model

def train(dataset_name):
    """
    Train the Barlow Twins model on a specified dataset.

    Args:
        dataset_name (str): Name of the dataset to be used for training.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = BarlowTwinsPretrainDataset(path=f'./data/processed/{dataset_name}-train-entities.pkl.gz',
                                               dataset=dataset_name)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    
    model = BarlowTwinsPretrainModel(len_tokenizer=len(train_dataset.tokenizer))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    criterion = BarlowTwins(lambd=1/4096, model_dim=768)
    criterion = criterion.to(device)

    for epoch in range(20):
        print('EPOCH: ', epoch)
        model = training_step(model=model,
                              device=device,
                              train_loader=train_loader,
                              optimizer=optimizer,
                              criterion=criterion)

    torch.save(model.state_dict(), f'./blockers/barlowtwins/saved_blockers/{dataset_name}/barlowtwins.pt')

if __name__ == '__main__':
    dataset_names = ['walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair']
    for dataset_name in dataset_names:
        train(dataset_name)
