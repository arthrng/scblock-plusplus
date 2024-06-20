import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from torch.nn import BCELoss
from model import DittoModel
from dataset import DittoDataset

def training_step(model, device, train_loader, optimizer, criterion, alpha):
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
    for (i, batch) in enumerate(train_loader):
        x1, x2, y = batch

        # Get outputs
        outputs = model(input_ids=x1.to(device),
                        input_ids_aug=x2.to(device))
        
        # Compute loss
        loss = criterion(outputs.view(-1, 1), y.to(device))

        # SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute running loss
        running_loss += loss.item() * len(y)
    
    # Compute training loss
    train_loss = running_loss / len(train_loader.dataset)
    print(train_loss)
    
    return model

def validation_step(model, device, val_loader, criterion):
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

    # Evaluate the model
    #num_correct = 0
    num_correct = []
    all_labels = []
    model.eval()

    # Set the running loss to 0
    running_loss = 0
    with torch.no_grad():
       for (i, batch) in enumerate(val_loader):
        x1, x2, y = batch

        # Get outputs
        outputs = model(input_ids=x1.to(device),
                        input_ids_aug=x2.to(device))
        
        # Compute loss
        y = y.to(device)
        outputs = outputs.to(device)
        loss = criterion(outputs.view(-1), y.float())

        num_correct += list((outputs.view(-1, 1) > 0.5) == y.view(-1, 1))
        all_labels += list(y.view(-1, 1))

        # Compute running loss
        running_loss += loss.item() * len(y)
    
    # Compute training loss
    val_loss = running_loss / len(val_loader.dataset)
    #print(val_loss)

    #print(num_correct / num_samples)
    num_correct = [int(t.cpu()) for t in num_correct]
    all_labels = [int(t.cpu()) for t in all_labels]
    print(num_correct)
    print(all_labels)
    print('F1: ', f1_score(all_labels, num_correct))
    
    return model


def train(dataset_name, alpha):
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
    train_dataset = DittoDataset(path=f'./data/processed/{dataset_name}-train-pairs.json.gz',
                                 dataset=dataset_name)
    val_dataset = DittoDataset(path=f'./data/processed/{dataset_name}-val-pairs.json.gz',
                                dataset=dataset_name)

    # Create data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=train_dataset.get_sampler(), collate_fn=train_dataset.pad)
    val_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=train_dataset.get_sampler(), collate_fn=train_dataset.pad)

    # Initialize model
    model = DittoModel(len_tokenizer=len(train_dataset.tokenizer), alpha=alpha)
    model.load_state_dict(torch.load( f'./matchers/ditto/saved_matchers/{dataset_name}/ditto-{alpha}.pt',
                                map_location=torch.device(device)), 
                                strict=False)
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    # Initialize criterion
    criterion = nn.BCELoss()
    criterion = criterion.to(device)

    # Train the model
    for epoch in range(1):
        # Show debug message
        print('EPOCH: ', epoch)
      
        # Perform training step
        # model = training_step(model=model,
        #                       device=device,
        #                       train_loader=train_loader,
        #                       optimizer=optimizer,
        #                       criterion=criterion,
        #                       alpha=alpha)
        validation_step(model=model,
                        device=device,
                        val_loader=val_loader,
                        criterion=criterion)

        

    # Save the model
    #torch.save(model.state_dict(), f'./matchers/ditto/saved_matchers/{dataset_name}/ditto-{alpha}.pt')

if __name__ == '__main__':
    # Initialize dataset names
    dataset_names = ['amazon-google'] #'walmart-amazon',

    # Initialize alphas
    alphas = list(np.round(np.arange(0.1, 1.0, 0.1), 2))
    for dataset_name in dataset_names:
      print(dataset_name)
      for alpha in alphas:
        print(alpha)
        train(dataset_name, alpha=alpha)