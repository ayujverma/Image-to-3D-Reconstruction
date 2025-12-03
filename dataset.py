# dataset.py
import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# -----------------------------
# Utility: BINVOX file loader
# -----------------------------
# Minimal binvox parser (works for ShapeNet R2N2)
def read_binvox(path):
    with open(path, 'rb') as f:
        # Read header
        line = f.readline().strip()
        if not line.startswith(b'#binvox'):
            raise IOError('Not a binvox file')

        dims = None
        depth = None
        height = None
        width = None
        translate = None
        scale = None

        # Parse header lines
        line = f.readline().strip()
        while line:
            if line.startswith(b'dim'):
                _, d, h, w = line.split()
                dims = (int(d), int(h), int(w))
            elif line.startswith(b'translate'):
                translate = [float(x) for x in line.split()[1:]]
            elif line.startswith(b'scale'):
                scale = float(line.split()[1])
            elif line.startswith(b'data'):
                break
            line = f.readline().strip()

        # Now read voxel data using run-length encoding
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        values = raw_data[0::2]
        counts = raw_data[1::2]

        voxels = np.zeros(np.prod(dims), dtype=np.uint8)
        idx = 0
        for v, c in zip(values, counts):
            voxels[idx: idx + c] = v
            idx += c

        voxels = voxels.reshape(dims)
        return voxels.astype(np.float32)  # 0/1 float values


# -----------------------------------------------------
# R2N2 / ShapeNet dataset loader
# -----------------------------------------------------
class R2N2Dataset(Dataset):
    def __init__(
        self,
        root,
        categories=None,
        views_per_model=1,
        transform=None,
        voxel_transform=None
    ):
        """
        Args:
            root: Root folder containing ShapeNetRendering/ and ShapeNetVoxels/
            categories: List of synset IDs (e.g., ["02691156"]) or None = use all
            views_per_model: Number of camera views to load (1–24)
            transform: torchvision transforms for images
            voxel_transform: optional transforms on voxel grids
        """
        self.root = root
        self.render_root = os.path.join(root, "ShapeNetRendering")
        self.voxel_root = os.path.join(root, "ShapeNetVoxels")

        self.transform = transform
        self.voxel_transform = voxel_transform
        self.views_per_model = views_per_model

        # Collect all models
        self.samples = []
        # dirs = ["03001627"]
        # print("Dirs found: ", dirs)
        # self.render_root = os.path.join(self.render_root, dirs[0])
        # self.voxel_root = os.path.join(self.voxel_root, dirs[0])
        cats = []
        if categories:
            cats = categories
        else:
            cats = ["03001627"]

        for cat in cats:
            render_cat_dir = os.path.join(self.render_root, cat)
            voxel_cat_dir = os.path.join(self.voxel_root, cat)

            if not os.path.isdir(render_cat_dir):
                continue
            max_num =0
            for model_id in sorted(os.listdir(render_cat_dir)):
                render_dir = os.path.join(render_cat_dir, model_id, "rendering")
                voxel_path = os.path.join(voxel_cat_dir, model_id, "model.binvox")

                if not os.path.isfile(voxel_path):
                    continue
                
                render_imgs = sorted(glob.glob(os.path.join(render_dir, "*.png")))

                if len(render_imgs) == 0:
                    continue

                self.samples.append({
                    "images": render_imgs,
                    "voxel": voxel_path
                })
                if max_num >32:
                    break
                max_num +=1

        print(f"[R2N2Dataset] Loaded {len(self.samples)} models.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img_paths = entry["images"]
        voxel_path = entry["voxel"]

        # ----------------------
        # Load VIEWS
        # ----------------------
        # Choose N views deterministically (first N for simplicity)
        chosen_views = img_paths[:self.views_per_model]

        imgs = []
        for p in chosen_views:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            imgs.append(img)

        # If single-view return tensor CxHxW, else VxCxHxW
        if self.views_per_model == 1:
            imgs = imgs[0]
        else:
            imgs = torch.stack(imgs, dim=0)

        # ----------------------
        # Load VOXELS
        # ----------------------
        vox = read_binvox(voxel_path)  # float32 0/1 array
        vox = torch.tensor(vox).unsqueeze(0)  # → 1xDxDxD

        if self.voxel_transform:
            vox = self.voxel_transform(vox)

        return {
            "images": imgs,
            "voxels": vox,
            "model_id": os.path.basename(os.path.dirname(voxel_path))
        }

def load_data():
    from dataset import R2N2Dataset
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    r2n2path = "/Users/maadhavkothuri/Documents/UT Austin Fall 2025/CS395T/A3/Image-to-3D-Reconstruction/dataset/r2n2_shapenet_dataset/r2n2"
    dataset = R2N2Dataset(
        root= r2n2path,
        views_per_model=1,
        transform=transform
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample data keys: {dataset[0].keys()}")
    print(f"Sample model ID: {dataset[0]['model_id']}")
    print(f"Image shape: {dataset[0]['images'].shape}")
    print(f"Voxel shape: {dataset[0]['voxels'].shape}")
    return dataset
load_data()