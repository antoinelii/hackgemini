from pathlib import Path
from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math

from baseline.dataset import BaselineDataset, get_train_val_Dataloaders
from unet_3d.model import UNet3D

TRAIN_FILEPATH = "/Users/33783/Desktop/capgemini/hackathon-mines-invent-2024/DATA/TRAIN"

# To adapt to mIoU need to put emphasis on less frequent classes
# calculated as normalized inverse of sqrt(frequency) in the train set
CLASS_WEIGHTS = torch.Tensor([0.0090, 0.0135, 0.0218, 0.0195, 0.0399, 0.0427, 0.0701, 0.0569, 0.0335,
        0.0618, 0.0688, 0.0551, 0.0585, 0.1052, 0.0449, 0.0566, 0.0582, 0.0791,
        0.0867, 0.0183])

def print_iou_per_class(
    targets: torch.Tensor,
    preds: torch.Tensor,
    nb_classes: int,
) -> None:
    """
    Compute IoU between predictions and targets, for each class.

    Args:
        targets (torch.Tensor): Ground truth of shape (B, H, W).
        preds (torch.Tensor): Model predictions of shape (B, nb_classes, H, W).
        nb_classes (int): Number of classes in the segmentation task.
    """

    # Compute IoU for each class
    # Note: I use this for loop to iterate also on classes not in the demo batch

    iou_per_class = []
    for class_id in range(nb_classes):
        iou = jaccard_score(
            targets == class_id,
            preds == class_id,
            average="binary",
            zero_division=0,
        )
        iou_per_class.append(iou)

    for class_id, iou in enumerate(iou_per_class):
        print(
            "class {} - IoU: {:.4f} - targets: {} - preds: {}".format(
                class_id, iou, (targets == class_id).sum(), (preds == class_id).sum()
            )
        )


def print_mean_iou(targets: torch.Tensor, preds: torch.Tensor) -> None:
    """
    Compute mean IoU between predictions and targets.

    Args:
        targets (torch.Tensor): Ground truth of shape (B, H, W).
        preds (torch.Tensor): Model predictions of shape (B, nb_classes, H, W).
    """

    mean_iou = jaccard_score(targets, preds, average="macro")
    print(f"meanIOU (over existing classes in targets): {mean_iou:.4f}")

def get_preds_from_raw_out(raw_out):
    return raw_out.max(dim=2).values.argmax(dim=1)

def preprocess_month_median(input_batch):
    """
    Here we want to use the collapse the temporal dimension of the input
    batch by keeping the median value for each pixel for each month
    going from [batch_size, T, 10, 128, 128]
    to [batch_size, 11, 8, 128, 128]
    (we select only 8 channels to match model architecture (needs power of 2))
    (We don't do december since not all samples have an image for this month)

    We also return a batch of vectors of size 3 
    that contains TILE, N_Parcel and Parcel_Cover info

    input_batch: dataloader X dict batch
    """
    batch_len = input_batch["date"].shape[0]

    batch_embeds = []
    batch_vectors = []
    for i in range(batch_len):
        median_imgs = []
        for month in range(1, 12):
            month_locs = []
            month_locs = torch.where(input_batch["date"][i,:,1] == month)[0]
            
            median = torch.median(input_batch["S2"][i,month_locs,:8], dim=0, keepdim=False)
            median_imgs.append(median[0])
            sample_embed = torch.stack(median_imgs, dim=0)
        batch_embeds.append(sample_embed)

    batch_vectors =torch.cat([input_batch["TILE"],
                              input_batch["N_Parcel"],
                              input_batch["Parcel_Cover"]], dim=1).to(torch.float32)
    
    return torch.stack(batch_embeds, dim=0), batch_vectors

def train_model(
    data_folder: Path,
    nb_classes: int,
    input_channels: int,
    model_path: None,
    train_ratio : float = 0.8,
    preprocess_batch_fun = preprocess_month_median,
    load_model: bool = False,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
) -> UNet3D:
    """
    Training pipeline.
    """
    dt_train = BaselineDataset(Path(data_folder))
    
    train_loader, val_loader = get_train_val_Dataloaders(
        dt_train,
        train_ratio=train_ratio,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        )
    
    print('Data Loaded Successfully!')

    epoch = 0 # epoch is initially assigned to 0. If load_model is true then
              # epoch is set to the last value + 1. 
    train_loss_values = [] # Defining a list to store loss values after every epoch
    val_loss_values = []

    # Defining the model, optimizer and loss function
    model = UNet3D(in_add_features=3, in_channels=input_channels , num_classes= nb_classes).to(device).train()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    # Loading a previous stored model from model_path variable
    if load_model == True:
        if device == "cuda":
            checkpoint = torch.load(model_path)
        elif device == "cpu":
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch'] + 1
        train_loss_values = checkpoint['train_loss_values']
        val_loss_values = checkpoint['val_loss_values']
        print("Model successfully loaded!")

    class_weights = CLASS_WEIGHTS.to(device)
    
    min_valid_loss = math.inf

    for e in range(num_epochs):
        
        train_loss = 0.0
        model.train()
        for i, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, inputs_vector = preprocess_batch_fun(inputs)
            targets = targets.unsqueeze(1).expand(-1, 8, -1,-1)
            inputs, inputs_vector, targets = inputs.to(device), inputs_vector.to(device), targets.to(device)
            
            outputs = model(inputs, inputs_vector)
            loss = criterion(outputs, targets.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        valid_loss = 0.0
        model.eval()
        for i, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, inputs_vector = preprocess_batch_fun(inputs)
            inputs, inputs_vector, targets = inputs.to(device), inputs_vector.to(device), targets.to(device)
            
            outputs = model(inputs, inputs_vector)
            loss = criterion(outputs,targets.long())
            valid_loss = loss.item()
        
        train_loss_values = train_loss_values.append(train_loss)
        val_loss_values = val_loss_values.append(valid_loss)
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')
        
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'epoch': e,
                'train_loss_values': train_loss_values,
                'val_loss_values': val_loss_values,
            }, f'unet3d_epoch{e}_valLoss{min_valid_loss}.pth')

    print("Training complete.")


if __name__ == "__main__":
    # Example usage:
    train_model(
        data_folder=TRAIN_FILEPATH,
        nb_classes=20,
        input_channels=11,
        model_path="models/unet3d.pth",
        preprocess_batch_fun=preprocess_month_median,
        load_model=True,
        num_epochs=2,
        batch_size=8,
        learning_rate=5e-4,
        device= "cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )