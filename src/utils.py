

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import os

ARTIFACTS_DIR = "artifacts"
# ---------- small helpers ----------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_model_dir(model_kind, base_dir = ARTIFACTS_DIR) :
    d = os.path.join(base_dir, model_kind.lower())
    ensure_dir(d)
    return d

def train_single_epoch(model, loader, optimizer, loss_fn, device):

    model.train()
    sum_loss, sum_correct, total_n = 0.0, 0.0, 0  #total_n is total number of samples. total_correct is total number of correct samples. total_loss is total loss of all samples.
    
    for image,label in loader:
        # data trsfer to GPU
        image=image.to(device)
        label=label.to(device)
        #clear gradients
        optimizer.zero_grad()
        #forward pass
        outputs=model(image)
        #loss calculation
        loss=loss_fn(outputs,label)
        #backward pass
        loss.backward()
        #Weights updation
        optimizer.step()

        sum_loss=sum_loss+loss.item()*label.size(0)
        sum_correct=sum_correct+(outputs.argmax(1)==label).sum().item()
        total_n=total_n+label.size(0)

        avg_loss=sum_loss/total_n
        avg_accuracy=sum_correct/total_n
    return avg_loss, avg_accuracy

@torch.no_grad()
def evaluate_valset(model, loader, loss_fn, device) :

    model.eval()
    sum_loss, sum_correct, total_n = 0.0, 0.0, 0  
    
    for image,label in loader:
        image=image.to(device)
        label=label.to(device)

        outputs=model(image)
        loss=loss_fn(outputs,label)

        sum_loss=sum_loss+loss.item()*label.size(0)
        sum_correct=sum_correct+(outputs.argmax(1)==label).sum().item()
        total_n=total_n+label.size(0)

        avg_loss=sum_loss/total_n
        avg_accuracy=sum_correct/total_n

    return avg_loss, avg_accuracy

def save_plots():
    mlp=pd.read_csv("artifacts/mlp_history.csv")
    cnn=pd.read_csv("artifacts/cnn_history.csv")
#plot 1 accuracy CNN vs MLP
    plt.figure(figsize=(8,5))
    plt.plot(mlp.epoch, mlp.train_acc, label="MLP Train", color="tab:blue", linewidth=2)
    plt.plot(mlp.epoch, mlp.val_acc,   label="MLP Val",   color="tab:blue", linestyle="--", linewidth=2)
    plt.plot(cnn.epoch, cnn.train_acc, label="CNN Train", color="tab:orange", linewidth=2)
    plt.plot(cnn.epoch, cnn.val_acc,   label="CNN Val",   color="tab:orange", linestyle="--", linewidth=2)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training and Validation Accuracy (MLP vs CNN)", fontsize=14, weight="bold")
    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1,0.5))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/accuracy_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

#plot 2 loss CNN vs MLP
    plt.figure(figsize=(8,5))
    plt.plot(mlp.epoch, mlp.train_loss, label="MLP Train", color="tab:blue", linewidth=2)
    plt.plot(mlp.epoch, mlp.val_loss,   label="MLP Val",   color="tab:blue", linestyle="--", linewidth=2)
    plt.plot(cnn.epoch, cnn.train_loss, label="CNN Train", color="tab:orange", linewidth=2)
    plt.plot(cnn.epoch, cnn.val_loss,   label="CNN Val",   color="tab:orange", linestyle="--", linewidth=2)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss (MLP vs CNN)", fontsize=14, weight="bold")
    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1,0.5))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

def save_batch_plot(): 
    batch_mlp=pd.read_csv("artifacts/mlp_batch/batch_runtime.csv")
    batch_cnn=pd.read_csv("artifacts/cnn_batch/batch_runtime.csv")

    plt.figure(figsize=(8,5))
    plt.plot(batch_mlp.batch_size, batch_mlp.runtime, label="MLP", 
            color="tab:blue", marker="o", linewidth=2)
    plt.plot(batch_cnn.batch_size, batch_cnn.runtime, label="CNN", 
            color="tab:orange", marker="s", linewidth=2)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Runtime (s)", fontsize=12)
    plt.title("Runtime vs Batch Size (MLP vs CNN)", fontsize=14, weight="bold")
    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1,0.5))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/batch_runtime.png", dpi=300, bbox_inches="tight")
    plt.close()
