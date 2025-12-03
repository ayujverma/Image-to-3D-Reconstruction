# voxel_decoder.py
import torch
import torch.nn as nn
from encoder.encoder import ImageEncoder
from dataset import R2N2Dataset, load_data
from torch.utils.data import DataLoader


class VoxelDecoder(nn.Module):
    """
    Decoder that maps a latent vector z → voxel occupancy grid.
    Uses an MLP to map z to a small 3D feature grid, then 3D deconvolutions
    to upsample to the final voxel resolution.
    """

    def __init__(self, latent_dim=512, voxel_resolution=32, base_channels=64):
        """
        Args:
            latent_dim: dimension of encoder output (e.g., 512 for ResNet18)
            voxel_resolution: final voxel grid size (e.g., 32, 64)
            base_channels: number of channels for initial 3D feature volume
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.voxel_resolution = voxel_resolution
        self.base_channels = base_channels

        # ----------------------------------------------------
        # Step 1: Map latent vector to a small 3D feature cube
        # ----------------------------------------------------
        # We'll grow from 4×4×4 → final resolution.
        self.init_res = 4

        init_volume = base_channels * (self.init_res**3)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, init_volume),
            nn.ReLU(inplace=True)
        )

        # This reshapes to [B, C, 4, 4, 4]
        # Channels decrease as we upsample.
        # Upsampling stages depend on desired final resolution.
        # 4 → 8 → 16 → 32 (three stages for 32)
        conv_layers = []
        current_res = self.init_res
        current_channels = base_channels

        while current_res < voxel_resolution:
            next_channels = current_channels // 2 if current_channels > 8 else current_channels

            conv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_channels=current_channels,
                        out_channels=next_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm3d(next_channels),
                    nn.ReLU(inplace=True)
                )
            )

            current_res *= 2
            current_channels = next_channels

        # Final 1x1x1 conv to produce occupancy values
        self.deconv = nn.Sequential(
            *conv_layers,
            nn.Conv3d(current_channels, 1, kernel_size=1),
            nn.Sigmoid()  # output occupancy probabilities ∈ [0,1]
        )


    def forward(self, z):
        """
        Args:
            z: latent vector [B, latent_dim]

        Returns:
            voxel grid [B, 1, D, D, D]
        """
        B = z.size(0)

        # Map latent vector → initial 3D feature cube
        x = self.fc(z)                                     # [B, C*4*4*4]
        x = x.view(B, self.base_channels,
                   self.init_res, self.init_res, self.init_res)

        # Upsample to target resolution
        x = self.deconv(x)                                 # [B,1,D,D,D]

        return x

class Image2VoxelModel(nn.Module):
    def __init__(self, latent_dim=512, voxel_res=32, pretrained_encoder=False):
        super().__init__()

        self.encoder = ImageEncoder(
            pretrained=pretrained_encoder,
            latent_dim=latent_dim
        )

        self.decoder = VoxelDecoder(
            latent_dim=latent_dim,
            voxel_resolution=voxel_res
        )

    def forward(self, images):
        """
        images:
            [B,3,H,W]  or
            [B,V,3,H,W]
        """
        z = self.encoder(images)         # → [B, latent_dim]
        vox = self.decoder(z)            # → [B,1,32,32,32]
        return vox



# -----------------------------------------
# Utility: IoU for voxel grids
# -----------------------------------------
def voxel_iou(pred, gt, threshold=0.5):
    """
    pred: [B,1,D,D,D]
    gt:   [B,1,D,D,D]
    """
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * gt).sum(dim=[1,2,3,4])
    union = ((pred_bin + gt) > 0).float().sum(dim=[1,2,3,4])
    iou = intersection / (union + 1e-6)
    return iou.mean().item()



# -----------------------------------------
# Training and validation loops
# -----------------------------------------
def train_one_epoch(model, dataloader, optimizer, device):
    
    total_loss = 0

    bce = nn.BCELoss()

    for batch in dataloader:
        imgs = batch["images"].to(device)
        vox_gt = batch["voxels"].to(device)

        optimizer.zero_grad()

        vox_pred = model.decoder(model.encoder(imgs))
        loss = bce(vox_pred, vox_gt)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    bce = nn.BCELoss()

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["images"].to(device)
            vox_gt = batch["voxels"].to(device)

            vox_pred = model(imgs)
            loss = bce(vox_pred, vox_gt)

            total_loss += loss.item()
            total_iou += voxel_iou(vox_pred, vox_gt)

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou
