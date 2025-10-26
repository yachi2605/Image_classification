# train.py
import time
from typing import List, Tuple
import torch
from torch import nn
from torch.optim import Optimizer

from src.utils import  evaluate_valset,train_single_epoch  # all “extras” live in utils.py
  
def train(model,train_loader,valid_loader,epochs= 10,lr = 0.001,device_str=None) :

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    #loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = []

    # Epoch loop
    for epoch_num in range(1, epochs + 1):
        start_time = time.time()

        #  training set (one pass = one epoch)
        train_loss, train_acc = train_single_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        #validation set 
        val_loss, val_acc = evaluate_valset(
            model=model,
            loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
        )

        # Time
        seconds = time.time() - start_time
        # Save a record in list
        data.append({
            "epoch": epoch_num,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "seconds": float(seconds),
        })
        
        # Print  summary
        print(f"Epoch {epoch_num}/{epochs}: "
             f"training_loss={train_loss:.2f}, training_accuracy={train_acc:.2f}, "
             f"validation_loss={val_loss:.2f}, validation_accuracy={val_acc:.2f}")

    return data
