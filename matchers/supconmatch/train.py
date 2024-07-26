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
from model import ContrastiveClassifierModel
from dataset import ContrastiveClassificationDataset
from torch.nn import BCELoss
from sklearn.metrics import f1_score, recall_score, precision_score
    
def training_step(model, device, train_loader, optimizer, criterion):
    # Update the parameters of the model
    model.train()
    
    num_ones = 0
    num_zeroes = 0

    # Set the running loss to 0
    running_loss = 0
    for (batch1, batch2, labels) in train_loader:
        # Load first batch to device
        input_ids_left = batch1['input_ids'].to(device)
        attention_mask_left = batch1['attention_mask'].to(device)

        # Load second batch to device
        input_ids_right = batch2['input_ids'].to(device)
        attention_mask_right = batch2['attention_mask'].to(device)
        labels = labels.to(device)

        # Get outputs
        outputs = model(input_ids_left=input_ids_left, 
                        attention_mask_left=attention_mask_left,
                        input_ids_right=input_ids_right, 
                        attention_mask_right=attention_mask_right)

        num_ones += labels.sum()
        num_zeroes += len(labels) - labels.sum()
        
        # Compute loss
        loss = criterion(outputs, labels.view(-1, 1).float())

        # SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # Compute running loss
        running_loss += loss.item() * len(labels)
    
    print(num_ones, num_zeroes)
    
    # Compute training loss
    train_loss = running_loss / len(train_loader.dataset)
    print(train_loss)
    
    return model

def validation_step(model, device, val_loader, criterion):
    # Evaluate the model
    #num_correct = 0
    num_correct = []
    all_labels = []
    model.eval()

    # Set the running loss to 0
    running_loss = 0
    with torch.no_grad():
        for (batch1, batch2, labels) in val_loader:
            # Load first batch to device
            input_ids_left = batch1['input_ids'].to(device)
            attention_mask_left = batch1['attention_mask'].to(device)

            # Load second batch to device
            input_ids_right = batch2['input_ids'].to(device)
            attention_mask_right = batch2['attention_mask'].to(device)
            labels = labels.to(device)

            # Get outputs
            outputs = model(input_ids_left=input_ids_left, 
                            attention_mask_left=attention_mask_left,
                            input_ids_right=input_ids_right, 
                            attention_mask_right=attention_mask_right)
            
            # Compute loss
            #list((outputs > 0.5) == labels.view(-1, 1)).sum()
            num_correct += list(outputs > 0.5)
            all_labels += list(labels.view(-1, 1))

            #num_samples += len(labels)
            loss = criterion(outputs, labels.view(-1, 1).float())

            # Compute running loss
            running_loss += loss.item() * len(labels)
    
    # Compute training loss
    val_loss = running_loss / len(val_loader.dataset)
    print(val_loss)

    #print(num_correct / num_samples)
    num_correct = [int(t.item()) for t in num_correct]
    all_labels = [int(t.item()) for t in all_labels]

    print(sum(num_correct), sum(all_labels))
    print('F1: ', f1_score(all_labels, num_correct))
    print('Recall: ', recall_score(all_labels, num_correct))
    print('Precision: ', precision_score(all_labels, num_correct))

    return model

def train(dataset_name):

    # Initialize device
    device = torch.device("cuda:{}".format('0') if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    train_dataset = ContrastiveClassificationDataset(path=f'./data/processed/{dataset_name}-train-pairs.json.gz',
                                                     dataset=dataset_name)
    val_dataset = ContrastiveClassificationDataset(path=f'./data/processed/{dataset_name}-test-pairs.json.gz',
                                                   dataset=dataset_name)

    # Create data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=train_dataset.get_sampler()) 
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    model = ContrastiveClassifierModel(len_tokenizer=len(train_dataset.tokenizer))
    model.load_state_dict(torch.load(f'./blockers/scblock/saved_blockers/{dataset_name}/scblock.pt', map_location=torch.device(device)), strict=False)
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # Initialize criterion
    criterion = BCELoss()
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

        # Perform validation step
        validation_step(model=model,
                        device=device,
                        val_loader=val_loader,
                        criterion=criterion)

    # Save the model
    torch.save(model.state_dict(), f'./matchers/supcon-match/saved_matchers/{dataset_name}/supconmatch.pt')

if __name__ == '__main__':
  # Initialize dataset names
    dataset_names = ['walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair'] #'walmart-amazon',
    for dataset_name in dataset_names:
      train(dataset_name)