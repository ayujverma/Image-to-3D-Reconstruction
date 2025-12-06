# training_mesh.py (sketch)
import torch
from torch.utils.data import DataLoader
from encoder import ImageEncoder   # your encoder
from models import build_adjacency_matrix, Img2MeshModel, train_mesh_epoch
import numpy as np
from dataset import load_data
import json
import trimesh
import matplotlib.pyplot as plt

def train_mesh_model(model, train_loader, val_loader, optimizer, num_epochs=20, device="cpu", save_path=None):
    print("Starting training...")
    loss_dict = {"train_loss": [], "val_loss": []}
    for epoch in range(num_epochs):
        train_loss, val_loss = train_mesh_epoch(model, train_loader, val_loader, optimizer, device=device)
        loss_dict["train_loss"].append(train_loss)
        loss_dict["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if (epoch % 10 == 0 or epoch == num_epochs - 1) and save_path is not None:
            torch.save(model.state_dict(), save_path + f"epoch{epoch}.pth")
    return loss_dict

def visualize_image2mesh_results(model_path, dataset, meshes, template_mesh, index=0, device="cpu", save_path=None):
    image_tensor = dataset[index]["images"].to(device)  # [1, 3, H, W]
    gt_verts = meshes[index]["verts"].to(device)            # [V, 3]
    gt_faces = meshes[index]["faces"].to(device)            # [F,
    img = image_tensor.permute(1,2,0).cpu().numpy()
    with torch.no_grad():
        model = Img2MeshModel(template_verts_numpy=template_mesh["verts"], template_faces_numpy=template_mesh["faces"], adj_list=build_adjacency_matrix(template_mesh["faces"], template_mesh["verts"].shape[0]))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        pred_verts = model(image_tensor.unsqueeze(0))  # [1, V, 3]
        pred_verts = pred_verts.squeeze(0)  # [V, 3]

    # --- 2. Create meshes ---
    
    def plot_mesh_on_ax(ax, verts, faces, color='cyan', alpha=0.5):
        """
        Plot a triangular mesh on a given Matplotlib 3D axis.
        """
        # Loop over faces and plot triangle edges
        for f in faces:
            tri = verts[f]                   # [3, 3]
            tri = np.vstack([tri, tri[0]])   # close the triangle
            ax.plot(tri[:,0], tri[:,1], tri[:,2], color=color, alpha=alpha)
        
        ax.set_box_aspect([1,1,1])
        ax.axis('off')
    
    # gt_mesh = trimesh.Trimesh(vertices=gt_verts.cpu().numpy(),
    #                           faces=gt_faces.cpu().numpy(),
    #                           process=False)

    # pred_mesh = trimesh.Trimesh(vertices=pred_verts.cpu().numpy(),
    #                             faces=template_mesh["faces"],
                                # process=False)

    # --- 3. Render meshes ---
    # gt_render = gt_mesh.scene().save_image(resolution=(400,400))
    # pred_render = pred_mesh.scene().save_image(resolution=(400,400))

    # # Convert to displayable
    # gt_render = plt.imread(trimesh.util.wrap_as_stream(gt_render))
    # pred_render = plt.imread(trimesh.util.wrap_as_stream(pred_render))

    # --- 4. Plot side-by-side ---
    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    # Input image
    ax[0].imshow(img)
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    # Ground truth mesh
    ax[1] = fig.add_subplot(1,3,2, projection='3d')
    plot_mesh_on_ax(ax[1], gt_verts, gt_faces)
    ax[1].set_title("GT Mesh")

    # Predicted mesh
    ax[2] = fig.add_subplot(1,3,3, projection='3d')
    plot_mesh_on_ax(ax[2], pred_verts, template_mesh["faces"])
    ax[2].set_title("Predicted Mesh")

    if save_path is not None:
        plt.savefig(save_path + f"result_{index}.png")
    else:
        plt.show()
    

    # validation similar but without backward
def main():
    
    tpl = np.load("./img2mesh/template_icosphere.npz")
    adj_list = build_adjacency_matrix(tpl["faces"], tpl["verts"].shape[0])
    model = Img2MeshModel(template_verts_numpy=tpl["verts"], template_faces_numpy=tpl["faces"], adj_list=adj_list)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    save_path = "./img2mesh/saved_models/"
    trainset, testset, train_meshes, test_meshes = load_data()
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    print("Using device:", device)

    num_epochs = 50
    loss_dict = train_mesh_model(model, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer, device=device, num_epochs=num_epochs, save_path = save_path)
    with open("./img2mesh/losses.json", "w") as f:
        json.dump(loss_dict, f, indent=4)
    
    visualized_idx = [0, 52, 112, 162, 200]
    # visualized_idx = [0, 1, 2, 3, 4]

    for idx in visualized_idx:
        visualize_image2mesh_results(save_path + f"epoch{num_epochs -1}.pth", testset, test_meshes, template_mesh=tpl, index=idx, device=device, save_path = "./img2mesh/results/")

if __name__ == "__main__":
    main()