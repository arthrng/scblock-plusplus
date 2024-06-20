"""
Author: Arthur Ning (558688an@eur.nl)
Date: April 30, 2024
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import ContrastiveModel, ContrastivePretrainModel
from dataset import AuxiliaryFinetuneDataset, RobertaFinetuneDataset
from loss import SupConLoss

                # # Combine the two batches
                # batch = {}
                # batch['input_ids'] = torch.cat((batch1['input_ids'], batch2['input_ids']), dim=0)
                # batch['attention_mask'] = torch.cat((batch1['attention_mask'], batch2['attention_mask']), dim=0)
                # batch['labels'] = torch.cat((batch1['labels'], batch2['labels']), dim=0)
    
def training_step(model, device, train_loader, optimizer, criterion, flood_type=None, flood_levels=None):
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
    for (batch1, batch2) in train_loader:
        # Load first batch to device
        input_ids_left = batch1['input_ids'].to(device)
        attention_mask_left = batch1['attention_mask'].to(device)
        labels = batch1['labels'].to(device)

        # Load second batch to device
        input_ids_right = batch2['input_ids'].to(device)
        attention_mask_right = batch2['attention_mask'].to(device)

        # Get outputs
        outputs = model(input_ids_left=input_ids_left, 
                        attention_mask_left=attention_mask_left,
                        input_ids_right=input_ids_right, 
                        attention_mask_right=attention_mask_right)
        
        # Get flood levels
        if flood_type == 'ada':
            # Get ids belonging the product offers
            ids = batch1['id'] + batch2['id']

            # Get the flood levels
            batch_flood_levels = []
            for _id in ids:
                batch_flood_levels.append(flood_levels[_id])
            
            #print(len(batch_flood_levels), batch_flood_levels)
        
        # Compute loss
        if flood_type == 'regular':
          loss = criterion(outputs, labels, flood_levels=torch.tensor(flood_levels).to(device))
        elif flood_type == 'ada':
          loss = criterion(outputs, labels, flood_levels=torch.tensor(batch_flood_levels).to(device))
        else:
          loss = criterion(outputs, labels)

        # SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # Compute running loss
        running_loss += loss.item() * len(labels)
    
    # Compute training loss
    train_loss = running_loss / len(train_loader.dataset)
    print(train_loss)
    
    return model

def train(model_name, dataset_name, flood_type=None, flood_levels=None, num_auxiliary_models=20, train_aux_networks=False):
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
    train_dataset = RobertaFinetuneDataset(entities_path=f'./data/processed/{dataset_name}-train-entities.pkl.gz',
                                           pairs_path=f'./data/processed/{dataset_name}-train-pairs.json.gz',
                                           dataset=dataset_name)
    
    # Obtain auxiliary networks and compute flood levels
    if flood_type == 'ada' and train_aux_networks:
        # Initialize flood levels
        #_flood_levels = {}

        # Get data
        features, labels, ids = train_dataset.get_features()
        for i in range(0, num_auxiliary_models):  
            _flood_levels = {}

            # Message for debugging
            print(f'TRAINING AUXILIARY NETWORK {i}')

            train_dataset_aux = AuxiliaryFinetuneDataset(features=features,
                                            labels=labels,
                                            ids=ids,
                                            num_auxiliary_model=i,
                                            num_auxiliary_models=num_auxiliary_models,
                                            dataset=dataset_name)

            # Train the auxiliary network
            train_dataset_aux = train_auxiliary(features=features,
                                                labels=labels,
                                                ids = ids,
                                                num_auxiliary_model=i,
                                                num_auxiliary_models=num_auxiliary_models,
                                                dataset_name=dataset_name)

            # Initialize auxiliary model
            tokenizer = AutoTokenizer.from_pretrained('roberta-base', additional_special_tokens=('[COL]', '[VAL]'))
            model = ContrastiveModel(len_tokenizer=len(tokenizer), 
                                      model='roberta-base',
                                      is_auxiliary=True).to(device)
            model.load_state_dict(torch.load(f'./blockers/scblock/saved_blockers/{dataset_name}/auxiliary-model-{i}.pt', 
                                  map_location=torch.device(device)), 
                                  strict=False)

            # Compute the flooding levels
            heldout_features, heldout_labels, heldout_ids = train_dataset_aux.get_heldout_data()
            heldout_features_batches = np.array_split(np.array(heldout_features), 5)
            heldout_labels_batches = np.array_split(np.array(heldout_labels), 5)
            heldout_ids_batches = np.array_split(np.array(heldout_ids), 5)

            for b, batch in enumerate(heldout_features_batches):
                # Encode features
                heldout_encodings = tokenizer(list(batch), truncation=True, padding=True)

                # Get labels
                _labels = torch.tensor(heldout_labels_batches[b]).to(device)

                # Get ids
                _ids = heldout_ids_batches[b]

                # Construct embeddings
                outputs = model(input_ids=torch.tensor(heldout_encodings['input_ids']).to(device), 
                                attention_mask=torch.tensor(heldout_encodings['attention_mask']).to(device))

                # Initialize criterion
                criterion = SupConLoss(temperature=0.07, is_auxiliary=True)
                criterion = criterion.to(device)

                # Compute loss
                losses = criterion(outputs, _labels)
 
                # Store the flooding levels
                for l, loss in enumerate(losses):
                    _flood_levels[_ids[l]] = loss.item()

            # Save flood levels
            df = pd.DataFrame(_flood_levels.items(), columns=['id', 'flood_level'])
            df = df.explode('flood_level')
            df.to_csv(f'./data/flood_levels/{dataset_name}-{i}.csv')
    
    if flood_type == 'ada':
      flood_levels = {}
      for i in range(0, num_auxiliary_models):
        # Import flood levels
        df = pd.read_csv(f'./data/flood_levels/{dataset_name}-{i}.csv').drop(columns='Unnamed: 0')
        for _id, flood_level in dict(zip(df['id'], df['flood_level'])).items():
          flood_levels[_id] = flood_level

    # Create data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    model = ContrastivePretrainModel(len_tokenizer=len(train_dataset.tokenizer))
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    # Initialize criterion
    criterion = SupConLoss(temperature=0.07, flood_type=flood_type)
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
                              criterion=criterion,
                              flood_type=flood_type,
                              flood_levels=flood_levels)

    # Save the model
    torch.save(model.state_dict(), f'./blockers/scblock/saved_blockers/{dataset_name}/{model_name}.pt')

def train_auxiliary(features, labels, ids, num_auxiliary_model, num_auxiliary_models, dataset_name):
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

    
    def get_aux_dataset():
        return train_dataset

    # Prepare dataset
    train_dataset = AuxiliaryFinetuneDataset(features=features,
                                            labels=labels,
                                            ids = ids,
                                            num_auxiliary_model=num_auxiliary_model,
                                            num_auxiliary_models=num_auxiliary_models,
                                            dataset=dataset_name)

    # Create data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    # Intialize device
    device = torch.device("cuda:{}".format('0') if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = ContrastivePretrainModel(len_tokenizer=len(train_dataset.tokenizer),
                                     is_auxiliary=True)
    model.load_state_dict(torch.load(f'./blockers/scblock/saved_blockers/{dataset_name}/scblock.pt', 
                          map_location=torch.device(device)), 
                          strict=False)

    # Run model on GPU
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # Initialize criterion
    criterion = SupConLoss(temperature=0.07)
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
    torch.save(model.state_dict(), f'./blockers/scblock/saved_blockers/{dataset_name}/auxiliary-model-{num_auxiliary_model}.pt')

    # Return dataset
    return train_dataset

    
if __name__ == '__main__':
    # Intialize flooding type
    flood_type = 'ada'

    # Initialize dataset names
    dataset_names = ['walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair']

    # Train the model
    for dataset_name in dataset_names:
      print(dataset_name)
      # Apply regular flooding
      if flood_type == 'regular':
          flood_levels = list(np.arange(0.005, 0.03, 0.005))
          for flood_level in flood_levels:
              print(flood_level)
              train(model_name=f'scblock-with-flooding-{flood_level}', 
                  dataset_name=dataset_name, 
                  flood_type=flood_type, 
                  flood_levels=flood_level)
      # Apply AdaFlood
      elif flood_type == 'ada':
          train(model_name='scblock-with-adaflood',
              dataset_name=dataset_name,
              flood_type=flood_type,
              train_aux_networks=True)
      # Apply no flooding
      else:
          train(model_name='scblock',
              dataset_name=dataset_name)