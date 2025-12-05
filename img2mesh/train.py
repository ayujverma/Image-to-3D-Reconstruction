# training_mesh.py (sketch)
import torch
from torch.utils.data import DataLoader
from encoder import ImageEncoder   # your encoder
from models import MeshDecoder, mesh_supervision_loss, build_vertex_adj_list, ChamferDistance, Img2MeshModel
import numpy as np
from dataset import load_data
import json


chamfer_fn = ChamferDistance()

for epoch in range(30):
    model.encoder.train(); model.decoder.train()
    train_loss = 0.0
    for batch in train_loader:
        imgs = batch["image"].to(device)
        gt_pts = batch["gt_points"].to(device)   # [B, P, 3]
        optimizer.zero_grad()

        z = model.encoder(imgs)                  # [B, latent_dim]
        pred_verts = model.decoder(z)            # [B, V, 3]

        loss, parts = mesh_supervision_loss(pred_verts, model.decoder.F0, gt_pts,
                                           adj_list, chamfer_weight=1.0, smooth_weight=1e-3,
                                           edge_weight=1e-3, use_pytorch3d=use_pytorch3d, chamfer_fn=chamfer_fn)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}")

def train_mesh_model(model, train_loader, val_loader, optimizer, num_epochs=20, device="cpu", save_path=None):
    pass

def visualize_image2mesh_results(model_path, dataset, index=0, device="cpu", save_path=None):
    # load template mesh
    tpl = np.load("template_icosphere.npz")
    V0 = tpl["verts"]
    F0 = tpl["faces"]
    adj_list = build_vertex_adj_list(F0, V0.shape[0])
    

    # validation similar but without backward
def main():
    model = Img2MeshModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    save_path = "./img2pointcloud/saved_models/"
    trainset, testset = load_data()
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    print("Using device:", device)

    num_epochs = 50
    loss_dict = train_mesh_model(model, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer, device=device, num_epochs=num_epochs, save_path = save_path)
    with open("./img2voxel/losses.json", "w") as f:
        json.dump(loss_dict, f, indent=4)
    
    visualized_idx = [0, 52, 112, 162, 200]
    for idx in visualized_idx:
        visualize_image2mesh_results(save_path + f"epoch{num_epochs -1}.pth", testset, index=idx, device=device, save_path = "./img2voxel/results/")

if __name__ == "__main__":
    main()