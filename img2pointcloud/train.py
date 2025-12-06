import torch
import torch.nn as nn
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from dataset import load_data
from models import Img2PointCloudModel, ChamferDistance
import numpy as np

def train_pointcloud_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs=20,
    device="cpu",
    save_path=None
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
    criterion = ChamferDistance()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            images = batch["images"].to(device)         # [B, 3, H, W]
            gt_points = batch["points"].to(device)     # [B, N, 3]

            optimizer.zero_grad()

            # ------------------------------------
            # Forward pass: image → latent → points
            # ------------------------------------
            pred_points = model(images)                # [B, N, 3]
            loss = criterion(pred_points, gt_points)

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
                images = batch["images"].to(device)
                gt_points = batch["points"].to(device)

                pred_points = model(images)
                loss = criterion(pred_points, gt_points)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} → Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        if (epoch % 10 == 0 or epoch == num_epochs - 1) and save_path is not None:
            torch.save(model.state_dict(), save_path + f"epoch{epoch}.pth")

    print("Training complete.")
    losses_dict = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    with open("pointcloud_training_losses.json", "w") as f:
        json.dump(losses_dict, f, indent=4) 
    return

def visualize_image2pointcloud_results(model_path, dataset, index=0, device="cpu", save_path = None):
    """
    model_path: path to saved .pth model
    dataset: Image2PointCloudDataset object
    index: which example to visualize
    """

    # ------------------------------
    # Load model
    # ------------------------------
    checkpoint = torch.load(model_path, map_location=device)
    model = Img2PointCloudModel()
    img = dataset[index]["images"]     # (3,H,W) or (V,3,H,W)
    point_gt = dataset[index]["points"] # (N,3)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # ------------------------------
    # Run model
    # ------------------------------
    img = img.unsqueeze(0).to(device)      # (1,3,H,W) or (1,V,3,H,W)
    point_gt = point_gt.cpu().numpy()      # (N,3) numpy

    with torch.no_grad():
        point_pred = model(img)            # (1,N,3)
        point_pred = point_pred.squeeze(0).cpu().numpy()  # (N,3) numpy

    # ------------------------------
    # Visualization
    # ------------------------------
    img = img.squeeze(0)          # [C, H, W]
    img = img.permute(1, 2, 0).cpu().numpy()  # (H,W,C) for visualization 
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # ----------- Compute shared axis limits -----------
    all_points = np.concatenate([point_gt, point_pred], axis=0)
    min_xyz = all_points.min(axis=0)
    max_xyz = all_points.max(axis=0)

    # ----------- Plotting -----------
    fig = plt.figure(figsize=(15, 5))

    # --- IMAGE ---
    ax0 = fig.add_subplot(131)
    ax0.imshow(img)
    ax0.axis("off")
    ax0.set_title("Input Image")

    # --- GROUND TRUTH POINT CLOUD ---
    ax1 = fig.add_subplot(132, projection='3d')
    ax1.scatter(point_gt[:, 0], point_gt[:, 1], point_gt[:, 2], c='b', s=1)
    ax1.set_title("Ground Truth")
    ax1.set_xlim(min_xyz[0], max_xyz[0])
    ax1.set_ylim(min_xyz[1], max_xyz[1])
    ax1.set_zlim(min_xyz[2], max_xyz[2])

    # --- PREDICTED POINT CLOUD ---
    ax2 = fig.add_subplot(133, projection='3d')
    ax2.scatter(point_pred[:, 0], point_pred[:, 1], point_pred[:, 2], c='r', s=1)
    ax2.set_title("Prediction")
    ax2.set_xlim(min_xyz[0], max_xyz[0])
    ax2.set_ylim(min_xyz[1], max_xyz[1])
    ax2.set_zlim(min_xyz[2], max_xyz[2])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path + f"result_{index}.png")
    else:
        plt.show()

def main():
    model = Img2PointCloudModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    save_path = "./img2pointcloud/saved_models/"
    trainset, testset = load_data()
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    print("Using device:", device)

    num_epochs = 50
    loss_dict = train_pointcloud_model(model, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer, device=device, num_epochs=num_epochs, save_path = save_path)
    with open("./img2pointcloud/losses.json", "w") as f:
        json.dump(loss_dict, f, indent=4)
    
    visualized_idx = [0, 52, 112, 162, 200]
    for idx in visualized_idx:
        visualize_image2pointcloud_results(save_path + f"epoch{num_epochs -1}.pth", testset, index=idx, device=device, save_path = "./img2pointcloud/results/")
    

if __name__ == "__main__":
    main()