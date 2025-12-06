# mesh_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import ImageEncoder
import numpy as np


class ChamferDistance(nn.Module):
    """Compute Chamfer Distance between two point clouds.
       Input shapes:
         pred: [B, Np, 3]
         gt:   [B, Ng, 3]
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        B, Np, _ = pred.shape
        Ng = gt.shape[1]
        pred_expand = pred.unsqueeze(2)     # [B, Np, 1, 3]
        gt_expand = gt.unsqueeze(1)         # [B, 1, Ng, 3]

        dist = torch.sum((pred_expand - gt_expand)**2, dim=3)

        # Pred → GT
        min_pred_to_gt, _ = torch.min(dist, dim=2)  # [B, Np]
        # GT → Pred
        min_gt_to_pred, _ = torch.min(dist, dim=1)  # [B, Ng]
        chamfer = min_pred_to_gt.mean(dim=1) + min_gt_to_pred.mean(dim=1)
        return chamfer.mean()

class MeshDecoder(nn.Module):
    """
    Predict per-vertex displacements deltaV for a fixed template mesh V0 (V0 shape = [V,3]).
    Input: latent z [B, d]
    Output: predicted vertices [B, V, 3]  (= V0 + deltaV)
    """

    def __init__(self, latent_dim, template_verts_numpy, template_faces_numpy, hidden_dim=512):
        """
        template_verts_numpy: (V,3) np.array
        template_faces_numpy: (F,3) np.array (int)
        """
        super().__init__()
        self.latent_dim = latent_dim

        # store template mesh as tensors (register buffer so they move with model.to(device))
        V0 = torch.from_numpy(template_verts_numpy.astype("float32"))  # (V,3)
        F0 = torch.from_numpy(template_faces_numpy.astype("int64"))    # (F,3)
        self.register_buffer("V0", V0)   # [V,3], reader-only
        self.register_buffer("F0", F0)   # [F,3]

        V = V0.shape[0]

        # Simple MLP decoder predicting V * 3 coords
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, V * 3)
        )

    def forward(self, z):
        """
        z: [B, latent_dim]
        returns: pred_verts: [B, V, 3]
        """
        B = z.shape[0]
        out = self.mlp(z)                 # [B, V*3]
        out = out.view(B, -1, 3)         # [B, V, 3] --> predicted deltaV
        pred_verts = torch.add(self.V0.unsqueeze(0), out)  # broadcast add: [B, V, 3]
        return pred_verts                 # predicted mesh vertices

class Img2MeshModel(nn.Module):
    def __init__(self, latent_dim=512, pretrained_encoder=False,
                 template_verts_numpy=None, template_faces_numpy=None, hidden_dim=512, adj_list=None):
        super().__init__()
        self.encoder = ImageEncoder(
            pretrained=pretrained_encoder,
            latent_dim=latent_dim
        )
        self.decoder = MeshDecoder(
            latent_dim=latent_dim,
            template_verts_numpy=template_verts_numpy,
            template_faces_numpy=template_faces_numpy,
            hidden_dim=hidden_dim
        )
        self.register_buffer("adj_list", adj_list)

    def forward(self, images):
        """
        images: tensor of shape [B, 3, H, W] or [B, V, 3, H, W]
        returns: tensor of shape [B, V, 3] (predicted vertices)
        """
        z = self.encoder(images)        # shape [B, latent_dim]
        pred_verts = self.decoder(z)    # shape [B, V, 3]
        return pred_verts

def chamfer_loss(x, y):
    # x: [B, N, 3]
    # y: [B, M, 3]
    x_exp = x.unsqueeze(2)  # [B,N,1,3]
    y_exp = y.unsqueeze(1)  # [B,1,M,3]

    dist = torch.sum((x_exp - y_exp)**2, dim=3)  # [B,N,M]

    x2y = torch.min(dist, dim=2)[0]  # [B,N]
    y2x = torch.min(dist, dim=1)[0]  # [B,M]

    return x2y.mean() + y2x.mean()

def train_mesh_epoch(model, train_loader, test_loader, optimizer, device = "cpu"):
    """
    model: encoder + mesh decoder
    sample_fn: function(verts, faces, N) → sampled surface points   (your sampler)
    smoothness_fn: Laplacian or edge loss function
    """

    model.train()
    train_loss = 0
    val_loss = 0
    chamfer_fn = ChamferDistance()

    for batch in train_loader:
        imgs      = batch["images"].to(device)
        gt_points = batch["points"].to(device)   # pre-sampled GT mesh points
        faces     = model.decoder.F0.to(device)    # fixed template faces

        optimizer.zero_grad()

        # ---- Forward ----
        z = model.encoder(imgs)
        pred_verts = model.decoder(z)     # [B, V, 3]

        # Deform the template
        template_verts = model.decoder.V0.to(device)  # [V, 3]
        pred_verts = pred_verts + template_verts          # final mesh verts

        # ---- Sample points from predicted mesh ----
        pred_points = sample_points_on_mesh(pred_verts, faces, num_samples=2048)  # [B, N, 3]

        # ---- Chamfer Loss ----
        chamfer_loss_val = chamfer_fn(pred_points, gt_points)

        # ---- Smoothness Loss ----
        smooth_loss = laplacian_smoothness(pred_verts, model.adj_list)

        # Total loss
        loss = chamfer_loss_val + 0.2 * smooth_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        break
    avg_train_loss = train_loss / len(train_loader)
    
    for batch in test_loader:
        imgs      = batch["images"].to(device)
        gt_points = batch["points"].to(device)   # pre-sampled GT mesh points
        faces     = model.decoder.F0.to(device)    # fixed template faces

        with torch.no_grad():
            model.eval()
            # ---- Forward ----
            z = model.encoder(imgs)
            pred_verts = model.decoder(z)     # [B, V, 3]

            # Deform the template
            template_verts = model.decoder.V0.to(device)  # [V, 3]
            pred_verts = pred_verts + template_verts          # final mesh verts

            # ---- Sample points from predicted mesh ----
            pred_points = sample_points_on_mesh(pred_verts, faces, num_samples=2048)  # [B, N, 3]

            # ---- Chamfer Loss ----
            chamfer_loss = chamfer_fn(pred_points, gt_points)

            # ---- Smoothness Loss ----
            smooth_loss = laplacian_smoothness(pred_verts, model.adj_list) + edge_length_regularizer(pred_verts, faces)

            # Total loss
            loss = chamfer_loss + 0.2 * smooth_loss

            val_loss += loss.item()
            break
    avg_val_loss = val_loss / len(test_loader)

    return avg_train_loss, avg_val_loss


def build_adjacency_matrix(faces, num_vertices):
    """
    faces: [F, 3] long tensor
    returns:
       adj: [V, V] boolean adjacency matrix
    """
    if isinstance(faces, torch.Tensor):
        faces_np = faces.cpu().numpy()
    else:
        faces_np = faces

    adj = np.zeros((num_vertices, num_vertices), dtype=np.bool_)

    # For each triangle, add undirected edges
    for (a, b, c) in faces_np:
        adj[a, b] = adj[b, a] = True
        adj[a, c] = adj[c, a] = True
        adj[b, c] = adj[c, b] = True

    # Remove self-connections just in case
    np.fill_diagonal(adj, False)

    return torch.tensor(adj, dtype=torch.bool)



# -----------------------------------------------------
# 2. SAMPLE POINTS ON PREDICTED MESH (differentiable)
# -----------------------------------------------------
def sample_points_on_mesh(pred_verts, faces, num_samples):
        B, V, _ = pred_verts.shape
        device = pred_verts.device

        v0 = pred_verts[:, faces[:, 0], :]   # [B,F,3]
        v1 = pred_verts[:, faces[:, 1], :]
        v2 = pred_verts[:, faces[:, 2], :]
        # Areas for weighted sampling
        tri_areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=2), dim=2)  # [B,F]
        tri_probs = tri_areas / (tri_areas.sum(dim=1, keepdim=True) + 1e-9)

        # Sample triangles (not differentiable, but vertex ops ARE)
        with torch.no_grad():
            idx = torch.multinomial(tri_probs, num_samples, replacement=True)  # [B,Ng]

        # Gather triangle vertices
        v0_s = v0.gather(1, idx.unsqueeze(-1).repeat(1,1,3))  # [B,Ng,3]
        v1_s = v1.gather(1, idx.unsqueeze(-1).repeat(1,1,3))
        v2_s = v2.gather(1, idx.unsqueeze(-1).repeat(1,1,3))

        # Barycentric sampling (differentiable)
        u = torch.sqrt(torch.rand(B, num_samples, 1, device=device))
        v = torch.rand(B, num_samples, 1, device=device)

        pts = (1 - u) * v0_s + u * (1 - v) * v1_s + u * v * v2_s
        return pts  # [B,Ng,3]


# -----------------------------------------------------
# 3. LAPLACIAN SMOOTHNESS (differentiable)
# -----------------------------------------------------
def laplacian_smoothness(pred_verts, adj_list):
    B, V, _ = pred_verts.shape
    loss = 0.0

    for v in range(V):
        neigh = adj_list[v]
        if len(neigh) == 0:
            continue
        neigh_pts = pred_verts[:, neigh, :].mean(dim=1)  # [B,3]
        center = pred_verts[:, v, :]                     # [B,3]
        loss = loss + ((center - neigh_pts)**2).sum(dim=1).mean()

    return loss / V


# -----------------------------------------------------
# 4. EDGE REGULARIZER (discourages degenerate triangles)
# -----------------------------------------------------
def edge_length_regularizer(pred_verts, faces):
    v0 = pred_verts[:, faces[:, 0], :]
    v1 = pred_verts[:, faces[:, 1], :]
    v2 = pred_verts[:, faces[:, 2], :]

    e01 = torch.norm(v0 - v1, dim=2)
    e12 = torch.norm(v1 - v2, dim=2)
    e20 = torch.norm(v2 - v0, dim=2)

    mean_e = (e01 + e12 + e20) / 3.0
    reg = ((e01 - mean_e)**2 + (e12 - mean_e)**2 + (e20 - mean_e)**2).mean()
    return reg


# -----------------------------------------------------
# 5. FINAL LOSS (fully differentiable)
# -----------------------------------------------------
def mesh_supervision_loss(pred_verts, faces, gt_points,
                          adj_list,
                          w_surface=1.0,
                          w_smooth=1e-3,
                          num_samples=2048):
    """
    pred_verts: [B,V,3]
    faces: [F,3]
    gt_points: [B,Ng,3] (sampled GT surface pts)
    """

    # Surface matching term (Chamfer)
    pred_points = sample_points_on_mesh(pred_verts, faces, num_samples)
    surface_loss = chamfer_loss(pred_points, gt_points)

    # Smoothness term
    lap = laplacian_smoothness(pred_verts, adj_list)
    edge = edge_length_regularizer(pred_verts, faces)
    smooth_loss = lap + edge

    total = w_surface * surface_loss + w_smooth * smooth_loss
    return total

