from pathlib import Path
from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


from baseline.collate import pad_collate
from baseline.dataset import BaselineDataset
from unet.model import UNET

TRAIN_FILEPATH = "/Users/33783/Desktop/capgemini/hackathon-mines-invent-2024/DATA/TRAIN"

def preprocess_batch_august(input_batch):
    """
    Here we want to use the collapse the temporal dimension of the input
    batch by keeping only the first image of the month of august

    input_batch: dataloader X dict batch
    """
    L= []
    for i in range(input_batch["date"].shape[0]):
        L.append(torch.where(input_batch["date"][i,:,1] == 8)[0][0])
    indices_T_picked = torch.tensor(L)
    # Expand indices by creating a (B, 1, C, H, W) gather mask
    expanded_indices = indices_T_picked.view(-1, 1, 1, 1, 1).expand(-1, 1,
                                                                input_batch["S2"].size(2), input_batch["S2"].size(3),
                                                                input_batch["S2"].size(4))
    
    # Gather the values along the second dimension based on the indices
    collapsed_input_batch = torch.gather(input_batch["S2"], dim=1, index=expanded_indices).squeeze(1)

    return collapsed_input_batch

def train_model(
    data_folder: Path,
    nb_classes: int,
    input_channels: int,
    model_path: None,
    preprocess_batch_fun = preprocess_batch_august,
    load_model: bool = False,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
) -> UNET:
    """
    Training pipeline.
    """
    torch.manual_seed(1234)
    dt_train = BaselineDataset(Path(data_folder))

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        dt_train, batch_size=batch_size, collate_fn=pad_collate, shuffle=True
        )
    
    print('Data Loaded Successfully!')

    epoch = 0 # epoch is initially assigned to 0. If load_model is true then
              # epoch is set to the last value + 1. 
    loss_values = [] # Defining a list to store loss values after every epoch

    # Defining the model, optimizer and loss function
    model = UNET(in_channels=input_channels, classes=nb_classes).to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss() 

    # Loading a previous stored model from model_path variable
    if load_model == True:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch'] + 1
        loss_values = checkpoint['loss_values']
        print("Model successfully loaded!")

    #Training the model for every epoch.
    for e in range(epoch, num_epochs):
        print(f'Epoch: {e}')
        running_loss = 0.0
        for i, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = preprocess_batch_fun(inputs)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss = loss_function(outputs, targets.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss +=  loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        loss_values.append(epoch_loss) 
        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': e,
            'loss_values': loss_values
        }, model_path)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")
    print("Training complete.")


if __name__ == "__main__":
    # Example usage:
    train_model(
        data_folder=TRAIN_FILEPATH,
        nb_classes=20,
        input_channels=10,
        model_path="models/unet.pth",
        preprocess_batch_fun=preprocess_batch_august,
        load_model=True,
        num_epochs=10,
        batch_size=8,
        learning_rate=1e-3,
        device= "cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )