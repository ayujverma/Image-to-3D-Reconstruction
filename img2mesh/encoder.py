import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    A ResNet18-based encoder that maps 1 or more images to a latent feature vector z.
    Designed to be shared across voxel, point cloud, mesh, and implicit decoders.
    """

    def __init__(self, pretrained=False, latent_dim=512, multi_view_pool="mean"):
        """
        Args:
            pretrained: whether to load ImageNet-pretrained weights
            latent_dim: output feature dimension (512 for ResNet-18)
            multi_view_pool: how to combine multiple views ("mean", "max", or None)
        """
        super().__init__()
        net = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(net.children())[:-1])

        # Latent dimension of ResNet-18 is 512
        self.latent_dim = latent_dim

        # How to pool multi-view inputs
        self.multi_view_pool = multi_view_pool


    def forward(self, x):
        """
        Args:
            x: Either
               - Single image: [B,3,H,W]
               - Multi-view:   [B,V,3,H,W]
        Returns:
            z: [B,latent_dim]
        """

        # -------------------------
        # Case 1: MULTI-VIEW
        # -------------------------
        if x.dim() == 5:
            B, V, C, H, W = x.shape
            x = x.view(B * V, C, H, W)                 # merge batch and view dims
            feats = self.backbone(x)                   # → [B*V, 512, 1, 1]
            feats = feats.view(B, V, self.latent_dim)  # → [B, V, 512]

            if self.multi_view_pool == "mean":
                z = feats.mean(dim=1)
            elif self.multi_view_pool == "max":
                z = feats.max(dim=1).values
            else:
                z = feats  # [B, V, 512]

            return z

        else:
            feats = self.backbone(x)               # [B,512,1,1]
            feats = feats.view(x.size(0), -1)      # [B,512]
            return feats
