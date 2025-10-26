
from src.data import train_loader, valid_loader
from src.models import SimpleNN, CnnModel
from src.train import train
import pandas as pd
from src.utils import save_plots
from src.exp import run_all_experiments


EPOCHS = 10
LR = 0.001

def run(kind: str, model):
    print(f"\n=== Running {kind.upper()} for {EPOCHS} epochs (lr={LR}), device=cuda ===")   
    data = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=EPOCHS,
        lr=LR,
        device_str="cuda",   
    )
    pd.DataFrame.from_records(data).to_csv(f"artifacts/{kind}_history.csv", index=False)
    save_plots()

if __name__ == "__main__":
    # 1) MLP
    run("mlp", SimpleNN())
    # 2) CNN
    run("cnn", CnnModel(out_class=10))
    run_all_experiments()
  