import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import json

def plot_metrics(train_losses, val_losses):
    """
    Save loss plot and per-epoch data for frontend.
    """
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title("Train vs Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    save_path = assets_dir / "loss_plot.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    # Save JSON for frontend
    data = {
        "epochs": list(epochs),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    json_path = assets_dir / "loss_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    print(f"✅ Plot saved: {save_path}")
    print(f"✅ Data saved: {json_path}")

    return str(save_path), str(json_path)
