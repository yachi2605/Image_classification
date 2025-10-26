# exp.py
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import SGD
from src.data import make_loaders
from src.models import SimpleNN, CnnModel
from src.train import train
from src.utils import  get_model_dir,save_batch_plot

def experiment_batch_size(model_class, model_name, batch_sizes, epochs, lr):
    results = []
    for bs in batch_sizes:
        train_loader, valid_loader = make_loaders(batch_size=bs)
        model = model_class()
        t = time.time()
        data = train(model, train_loader, valid_loader, epochs=epochs, lr=lr)
        runtime = time.time() - t
        results.append({"batch_size": bs, "runtime": runtime})
    df = pd.DataFrame(results)
    output_dir = get_model_dir(model_name + "_batch")
    df.to_csv(f"{output_dir}/batch_runtime.csv", index=False)
    


def experiment_lr(model_class, model_name, lrs, epochs):
    results = []
    for lr in lrs:
        train_loader, valid_loader = make_loaders()
        model = model_class()
        t = time.time()
        data = train(model, train_loader, valid_loader, epochs=epochs, lr=lr)
        runtime = time.time() - t
        final_acc = data[-1]["val_acc"]
        results.append({"lr": lr, "val_acc": final_acc, "runtime": runtime})
    df = pd.DataFrame(results)
    output_dir = get_model_dir(model_name + "_lr")
    df.to_csv(f"{output_dir}/lr_acc.csv", index=False)
    df.plot(x="lr", y="runtime", marker="o", title=f"{model_name} – Runtime vs LR")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/lr_runtime.png", dpi=300)
    plt.close()

    df.plot(x="lr", y="val_acc", marker="o", title=f"{model_name} – Val Acc vs LR")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/lr_acc.png", dpi=300)
    plt.close()

def experiment_momentum(model_class, model_name, momenta, epochs):

    results = []
    train_loader, valid_loader = make_loaders()
    for m in momenta:
        model = model_class()
        device = "cuda" 
        model.to(device)
        opt = SGD(model.parameters(), lr=0.001, momentum=m)
        loss_fn = nn.CrossEntropyLoss()
        # quick 1-epoch test
        t = time.time()
        data = train(model, train_loader, valid_loader, epochs=epochs, lr=0.001)
        runtime = time.time() - t
        final_acc = data[-1]["val_acc"]
        results.append({"momentum": m, "runtime": runtime, "val_acc": final_acc})
    df = pd.DataFrame(results)
    out_dir = get_model_dir(model_name + "_momentum")
    # Acc vs momentum
    df.plot(x="momentum", y="val_acc", marker="o",title=f"{model_name} – Val Acc vs Momentum")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_dir}/momentum_acc.png", dpi=300)
    plt.close()

    # Runtime vs momentum
    df.plot(x="momentum", y="runtime", marker="o",title=f"{model_name} – Runtime vs Momentum")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_dir}/momentum_runtime.png", dpi=300)
    plt.close()

def run_all_experiments():
    for model_class, name in [(SimpleNN, "mlp"), (CnnModel, "cnn")]:
        experiment_batch_size(model_class, name, batch_sizes=[32, 64, 128, 256], epochs=10, lr=0.001)
        experiment_lr(model_class, name, lrs=[0.0001, 0.001, 0.01, 0.1], epochs=10)
        experiment_momentum(model_class, name, momenta=[0.0, 0.3, 0.5, 0.7, 0.9], epochs=10)


    save_batch_plot()