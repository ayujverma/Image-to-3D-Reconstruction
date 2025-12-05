import torch
from dataset import load_data
from models import Image2VoxelModel, train_one_epoch, validate
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

def train_model(model, train_loader = None, val_loader = None, optimizer = None, device = 'cpu', num_epochs = 30, save_path = None):
    train_losses, val_losses, val_ious = [], [], []
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device = "cpu")

        print(f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f} - ")
        train_losses.append(train_loss)

        if val_loader is not None:
                with torch.no_grad():
                    val_loss, val_iou = validate(model, val_loader, device)
                    print(f"Validation Loss: {val_loss:.4f} - Validation IoU: {val_iou:.4f}")
                    val_losses.append(val_loss)
                    val_ious.append(val_iou)
    
        if (epoch % 10 == 0 or epoch == num_epochs - 1) and save_path is not None:
            torch.save(model.state_dict(), save_path + f"epoch{epoch}.pth")
    loss_dict = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_ious": val_ious
    }
    return loss_dict


def visualize_voxel_grid(voxels, ax, title=""):
    """
    voxels: (D, D, D) numpy array with values in [0,1]
    """
    ax.voxels(voxels > 0.5, edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def visualize_image2voxel_results(model_path, dataset, index=0, device="cpu", model_type = "img2voxel", save_path = None):
    """
    model_path: path to saved .pth model
    dataset: R2N2Dataset object
    index: which example to visualize
    """

    # ------------------------------
    # Load model
    # ------------------------------
    checkpoint = torch.load(model_path, map_location=device)
    if model_type == "img2voxel":
        model = Image2VoxelModel()
        img = dataset[index]["image"]     # (1,3,H,W)
        voxel_gt = torch.from_numpy(dataset[index]["voxel"]) # (1,D,D,D)
        voxel_gt =  voxel_gt.squeeze()  # (D,D,D) numpy
    else:
        raise ValueError("Unsupported model type")
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # ------------------------------
    # Run model
    # ------------------------------
    with torch.no_grad():
        pred_voxels = model(img.unsqueeze(0).to(device))           # (1,1,D,D,D)
        pred_voxels = pred_voxels.squeeze().cpu().numpy()

    # ------------------------------
    # Plot
    # ------------------------------
    fig = plt.figure(figsize=(12, 4))

    # Input image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img.permute(1,2,0))
    ax1.set_title("Input Image")
    ax1.axis("off")

    # Predicted voxels
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    visualize_voxel_grid(pred_voxels, ax2, title="Predicted Voxel Grid")

    # Ground truth voxels
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    visualize_voxel_grid(voxel_gt, ax3, title="Ground Truth Voxel Grid")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + f"visualization-{index}.png")
    # plt.show()

def main():
    model = Image2VoxelModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    save_path = "./img2voxel/saved_models/"
    trainset, testset = load_data()
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    print("Using device:", device)

    num_epochs = 40
    loss_dict = train_model(model, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer, device=device, num_epochs=num_epochs, save_path = save_path)
    with open("./img2voxel/losses.json", "w") as f:
        json.dump(loss_dict, f, indent=4)
    
    print(len(testset))
    visualized_idx = [0, 52, 112, 162, 200]
    for idx in visualized_idx:
        visualize_image2voxel_results(save_path + f"epoch{num_epochs -1}.pth", testset, index=idx, device=device, save_path = "./img2voxel/results/")
    

if __name__ == "__main__":
    main()