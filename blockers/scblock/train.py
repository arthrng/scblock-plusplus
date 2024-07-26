import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import ContrastiveModel, ContrastivePretrainModel
from dataset import AuxiliaryFinetuneDataset, RobertaFinetuneDataset
from loss import SupConLoss

def training_step(model, device, train_loader, optimizer, criterion, flood_type=None, flood_levels=None):
    """
    Perform a training step for the contrastive model.

    Args:
        model (nn.Module): The model to be trained.
        device (torch.device): Device to perform training on.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (nn.Module): Loss function.
        flood_type (str, optional): Type of flooding ('regular' or 'ada'). Defaults to None.
        flood_levels (dict, optional): Flood levels for each sample. Defaults to None.

    Returns:
        nn.Module: The trained model.
    """
    model.train()
    running_loss = 0

    for (batch1, batch2) in train_loader:
        # Move data to device
        input_ids_left = batch1['input_ids'].to(device)
        attention_mask_left = batch1['attention_mask'].to(device)
        labels = batch1['labels'].to(device)
        input_ids_right = batch2['input_ids'].to(device)
        attention_mask_right = batch2['attention_mask'].to(device)

        # Get model outputs
        outputs = model(input_ids_left=input_ids_left,
                        attention_mask_left=attention_mask_left,
                        input_ids_right=input_ids_right,
                        attention_mask_right=attention_mask_right)

        # Handle flooding levels
        if flood_type == 'ada':
            ids = batch1['id'] + batch2['id']
            batch_flood_levels = [flood_levels.get(_id, 0) for _id in ids]

        # Compute loss
        if flood_type == 'regular':
            loss = criterion(outputs, labels, flood_levels=torch.tensor(flood_levels).to(device))
        elif flood_type == 'ada':
            loss = criterion(outputs, labels, flood_levels=torch.tensor(batch_flood_levels).to(device))
        else:
            loss = criterion(outputs, labels)

        # Optimize model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * len(labels)

    # Compute average training loss
    train_loss = running_loss / len(train_loader.dataset)
    print(f'Training Loss: {train_loss}')

    return model

def train(model_name, dataset_name, flood_type=None, flood_levels=None, num_auxiliary_models=20, train_aux_networks=False):
    """
    Train the contrastive model with optional auxiliary models and flooding.

    Args:
        model_name (str): Name for saving the model.
        dataset_name (str): Name of the dataset.
        flood_type (str, optional): Type of flooding ('regular' or 'ada'). Defaults to None.
        flood_levels (dict, optional): Flood levels for each sample. Defaults to None.
        num_auxiliary_models (int, optional): Number of auxiliary models. Defaults to 20.
        train_aux_networks (bool, optional): If True, train auxiliary networks. Defaults to False.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    train_dataset = RobertaFinetuneDataset(
        entities_path=f'./data/processed/{dataset_name}-train-entities.pkl.gz',
        pairs_path=f'./data/processed/{dataset_name}-train-pairs.json.gz',
        dataset=dataset_name
    )

    if flood_type == 'ada' and train_aux_networks:
        _flood_levels = {}

        features, labels, ids = train_dataset.get_features()
        for i in range(num_auxiliary_models):
            print(f'Training Auxiliary Network {i}')

            train_dataset_aux = AuxiliaryFinetuneDataset(
                features=features,
                labels=labels,
                ids=ids,
                num_auxiliary_model=i,
                num_auxiliary_models=num_auxiliary_models,
                dataset=dataset_name
            )

            train_auxiliary(
                features=features,
                labels=labels,
                ids=ids,
                num_auxiliary_model=i,
                num_auxiliary_models=num_auxiliary_models,
                dataset_name=dataset_name
            )

            tokenizer = AutoTokenizer.from_pretrained('roberta-base', additional_special_tokens=('[COL]', '[VAL]'))
            model = ContrastiveModel(len_tokenizer=len(tokenizer),
                                      model='roberta-base',
                                      is_auxiliary=True).to(device)
            model.load_state_dict(torch.load(
                f'./blockers/scblock/saved_blockers/{dataset_name}/auxiliary-model-{i}.pt',
                map_location=device
            ), strict=False)

            heldout_features, heldout_labels, heldout_ids = train_dataset_aux.get_heldout_data()
            heldout_features_batches = np.array_split(np.array(heldout_features), 5)
            heldout_labels_batches = np.array_split(np.array(heldout_labels), 5)
            heldout_ids_batches = np.array_split(np.array(heldout_ids), 5)

            for b, batch in enumerate(heldout_features_batches):
                heldout_encodings = tokenizer(list(batch), truncation=True, padding=True)
                _labels = torch.tensor(heldout_labels_batches[b]).to(device)
                _ids = heldout_ids_batches[b]
                outputs = model(
                    input_ids=torch.tensor(heldout_encodings['input_ids']).to(device),
                    attention_mask=torch.tensor(heldout_encodings['attention_mask']).to(device)
                )
                criterion = SupConLoss(temperature=0.07, is_auxiliary=True).to(device)
                losses = criterion(outputs, _labels)
                for l, loss in enumerate(losses):
                    _flood_levels[_ids[l]] = loss.item()

            df = pd.DataFrame(_flood_levels.items(), columns=['id', 'flood_level'])
            df.to_csv(f'./data/flood_levels/{dataset_name}-{i}.csv')

    if flood_type == 'ada':
        flood_levels = {}
        for i in range(num_auxiliary_models):
            df = pd.read_csv(f'./data/flood_levels/{dataset_name}-{i}.csv').drop(columns='Unnamed: 0')
            flood_levels.update(dict(zip(df['id'], df['flood_level'])))

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    model = ContrastivePretrainModel(len_tokenizer=len(train_dataset.tokenizer)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = SupConLoss(temperature=0.07, flood_type=flood_type).to(device)

    for epoch in range(20):
        print(f'EPOCH: {epoch}')
        model = training_step(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            flood_type=flood_type,
            flood_levels=flood_levels
        )

    torch.save(model.state_dict(), f'./blockers/scblock/saved_blockers/{dataset_name}/{model_name}.pt')

def train_auxiliary(features, labels, ids, num_auxiliary_model, num_auxiliary_models, dataset_name):
    """
    Train an auxiliary model for flooding computation.

    Args:
        features (list): List of features.
        labels (list): List of labels.
        ids (list): List of IDs.
        num_auxiliary_model (int): Index of the auxiliary model.
        num_auxiliary_models (int): Total number of auxiliary models.
        dataset_name (str): Name of the dataset.

    Returns:
        AuxiliaryFinetuneDataset: The dataset used for training the auxiliary model.
    """
    train_dataset = AuxiliaryFinetuneDataset(
        features=features,
        labels=labels,
        ids=ids,
        num_auxiliary_model=num_auxiliary_model,
        num_auxiliary_models=num_auxiliary_models,
        dataset=dataset_name
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ContrastivePretrainModel(len_tokenizer=len(train_dataset.tokenizer), is_auxiliary=True)
    model.load_state_dict(torch.load(
        f'./blockers/scblock/saved_blockers/{dataset_name}/scblock.pt',
        map_location=device
    ), strict=False)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = SupConLoss(temperature=0.07).to(device)

    for epoch in range(20):
        print(f'EPOCH: {epoch}')
        model = training_step(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion
        )

    torch.save(model.state_dict(), f'./blockers/scblock/saved_blockers/{dataset_name}/auxiliary-model-{num_auxiliary_model}.pt')

    return train_dataset

if __name__ == '__main__':
    flood_type = 'ada'
    dataset_names = ['walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair']

    for dataset_name in dataset_names:
        print(f'Dataset: {dataset_name}')
        if flood_type == 'regular':
            flood_levels = list(np.arange(0.005, 0.03, 0.005))
            for flood_level in flood_levels:
                print(f'Flood Level: {flood_level}')
                train(
                    model_name=f'scblock-with-flooding-{flood_level}',
                    dataset_name=dataset_name,
                    flood_type=flood_type,
                    flood_levels=flood_level
                )
        elif flood_type == 'ada':
            train(
                model_name='scblock-with-adaflood',
                dataset_name=dataset_name,
                flood_type=flood_type,
                train_aux_networks=True
            )
        else:
            train(
                model_name='scblock',
                dataset_name=dataset_name
            )
