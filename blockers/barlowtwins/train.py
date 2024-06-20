"""
Author: Arthur Ning (558688an@eur.nl)
Date: April 30, 2024
"""

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import BarlowTwinsPretrainModel
from dataset import BarlowTwinsPretrainDataset

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    # Barlow Twins: https://arxiv.org/abs/2103.03230
    # Source code: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    def __init__(self, lambd, model_dim):
        super().__init__()
        self.lambd = lambd
        self.model_dim = model_dim

        # projector - Think about how this can be integrated!
        # projector
        sizes = [self.model_dim, 4096]  # + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, features):
        # features = self.projector(features)
        z1 = self.projector(features[:, 0, :])
        z2 = self.projector(features[:, 1, :])

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() * (1.0 / 256)
        off_diag = off_diagonal(c).pow_(2).sum() * (1.0 / 256)
        # print('On Diagonal: {}'.format(on_diag))
        # print('Off Diagonal: {}'.format(off_diag))
        loss = on_diag + self.lambd * off_diag

        return loss

def training_step(model, device, train_loader, optimizer, criterion):
    """This method retrieve the models words from the titles and the key-value pairs.
    
    Parameters
    ----------
    products: list
        list containing the data on the products
    
    Returns
    -------
    set
        set containing the model words
    """

    # Update the parameters of the model
    model.train()

    # Set the running loss to 0
    running_loss = 0

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', additional_special_tokens=('[COL]', '[VAL]'))
    for (batch1, batch2) in train_loader:
        # Load first batch to device
        batch1 = tokenizer(list(batch1), truncation=True, padding=True) 
        batch2 = tokenizer(list(batch2), truncation=True, padding=True)

        input_ids_left = torch.tensor(batch1['input_ids']).to(device)
        attention_mask_left = torch.tensor(batch1['attention_mask']).to(device)
        input_ids_right = torch.tensor(batch2['input_ids']).to(device)
        attention_mask_right = torch.tensor(batch2['attention_mask']).to(device)

        # Get outputs
        outputs = model(input_ids_left=input_ids_left, 
                        attention_mask_left=attention_mask_left,
                        input_ids_right=input_ids_right, 
                        attention_mask_right=attention_mask_right)
        
        # Compute loss
        loss = criterion(outputs)

        # SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute running loss
        running_loss += loss.item() * len(batch1)
    
    # Compute training loss
    train_loss = running_loss / len(train_loader.dataset)
    print(train_loss)
    
    return model

def train(dataset_name):
    """This method retrieve the models words from the titles and the key-value pairs.
    
    Parameters
    ----------
    products: list
        list containing the data on the products
    
    Returns
    -------
    set
        set containing the model words
    """

    # Initialize device
    device = torch.device("cuda:{}".format('0') if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    train_dataset = BarlowTwinsPretrainDataset(path=f'./data/processed/{dataset_name}-train-entities.pkl.gz',
                                               dataset=dataset_name)

    # Create data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = BarlowTwinsPretrainModel(len_tokenizer=len(train_dataset.tokenizer))
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # Initialize criterion
    criterion = BarlowTwins(lambd=1/4096, model_dim=768)
    criterion = criterion.to(device)

    # Train the model
    for epoch in range(20):
        # Show debug message
        print('EPOCH: ', epoch)
      
        # Perform training step
        model = training_step(model=model,
                               device=device,
                               train_loader=train_loader,
                               optimizer=optimizer,
                               criterion=criterion)

    # Save the model
    torch.save(model.state_dict(), f'./blockers/barlowtwins/saved_blockers/{dataset_name}/barlowtwins.pt')

if __name__ == '__main__':
    dataset_names = ['wdcproducts20pair', 'wdcproducts80pair']
    for dataset_name in dataset_names:
      train(dataset_name)