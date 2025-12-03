import torch
from encoder.encoder import ImageEncoder
from dataset import R2N2Dataset, load_data
from img2voxel.models import Image2VoxelModel, train_one_epoch
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def train_model(model, train_loader = None, val_loader = None, optimizer = None, device = 'cpu', num_epochs = 30, save_path = None):
    dataset = load_data()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device = "cpu")
        # val_loss, val_iou = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f} - ")
            # f"Val Loss: {val_loss:.4f} - "
            # f"Val IoU: {val_iou:.4f}")
        if epoch % 10 == 0 and save_path is not None:
            torch.save(model.state_dict(), save_path + f"epoch{epoch}-loss-{train_loss}.pth")

def visualize_voxel_grid(voxels, ax, title=""):
    """
    voxels: (D, D, D) numpy array with values in [0,1]
    """
    ax.voxels(voxels > 0.5, edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def visualize_image2voxel(model_path, dataset, index=0, device="cpu", model_type = "img2voxel"):
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
        img = dataset[index]["images"]     # (1,3,H,W)
        voxel_gt = dataset[index]["voxels"].squeeze().numpy()
        model_id = dataset[index]["model_id"]
    else:
        raise ValueError("Unsupported model type")
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # ------------------------------
    # Load a sample from the dataset
    # ------------------------------
    # img, voxel_gt, model_id = dataset[index]
    # print(dataset[index])
    # img = img.unsqueeze(0).to(device)      # (1,3,H,W)
    # voxel_gt = voxel_gt.squeeze().numpy()  # (D,D,D) numpy

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
    plt.show()

model = Image2VoxelModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
save_path = "./img2voxel/saved_models/"
dataset = load_data()
# train_model(model, optimizer=optimizer, device='cpu', num_epochs=30, save_path = save_path)
visualize_image2voxel(save_path + "epoch20-loss-0.3550262749195099.pth", dataset, index=0, device="cpu")