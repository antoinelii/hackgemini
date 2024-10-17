import torch
import torch.nn as nn
import torch.nn.functional as F

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(mIoULoss, self).__init__()

    def forward(self, preds, targets, nb_class=20, smooth=1):
        """
        Compute IoU between predictions and targets, for each class.

        Args:
            targets (torch.Tensor): Ground truth of shape (B, H, W).
            preds (torch.Tensor): Model predictions of shape (B, H, W).
            nb_classes (int): Number of classes in the segmentation task.
        """
        # Initialize IoU for each class
        iou_per_class = torch.zeros(nb_class)

        # Loop through each class and calculate IoU
        for cls in range(nb_class):
            pred_class = (preds == cls).float()
            target_class = (targets == cls).float()
            #flatten label and prediction tensors
            pred_class = pred_class.view(-1)
            target_class = target_class.view(-1)
            
            #intersection is equivalent to True Positive count
            #union is the mutually inclusive area of all labels & predictions 
            intersection = (pred_class * target_class).sum()
            total = (pred_class + target_class).sum()
            union = total - intersection 
            
            iou_per_class[cls] = (intersection + smooth)/(union + smooth)
        
        mIoU = iou_per_class.mean()

        return 1 - mIoU

# Example usage in training
"""
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_data)

        # Convert logits to predicted class indices
        preds = torch.argmax(outputs, dim=1)

        # Compute mIOU loss
        loss = mIoULoss(preds, batch_labels, num_classes=20)

        # Backprop and optimize
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
"""