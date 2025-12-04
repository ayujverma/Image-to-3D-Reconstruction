import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
import json

def train_pointcloud_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs=20,
    device="cpu",
    use_pytorch3d=True
):
    """
    model: nn.Module containing encoder + decoder
    train_loader: DataLoader returning {"image":..., "points":...}
    val_loader: same structure for validation
    optimizer: torch optimizer
    use_pytorch3d: True = use pytorch3d chamfer_distance
    chamfer_fn: ChamferDistance() if not using pytorch3d
    """

    model = model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            images = batch["image"].to(device)         # [B, 3, H, W]
            gt_points = batch["points"].to(device)     # [B, N, 3]

            optimizer.zero_grad()

            # ------------------------------------
            # Forward pass: image → latent → points
            # ------------------------------------
            pred_points = model(images)                # [B, N, 3]
            loss, _ = chamfer_distance(pred_points, gt_points)

            # Backprop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                gt_points = batch["points"].to(device)

                pred_points = model(images)
                loss, _ = chamfer_distance(pred_points, gt_points)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} → Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    print("Training complete.")
    losses_dict = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    with open("pointcloud_training_losses.json", "w") as f:
        json.dump(losses_dict, f, indent=4) 
    return