import numpy as np
import torch
from skimage import measure
import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json

def voxel_to_mesh(vox):
    """
    vox: numpy array [D, D, D] with 0/1 occupancy
    Returns (verts, faces)
    """
    verts, faces, normals, values = measure.marching_cubes(
        vox, 
        level=0.5, 
        spacing=(1.0, 1.0, 1.0)
    )
    return verts, faces

def sample_points_from_mesh(verts, faces, num_points=2048):
    """
    verts: (V, 3)
    faces: (F, 3) vertex indices
    Returns: Tensor [N, 3]
    """

    # Triangle vertices
    v0 = verts[faces[:,0]]
    v1 = verts[faces[:,1]]
    v2 = verts[faces[:,2]]

    # Triangle areas (cross product magnitude / 2)
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2
    area_sum = np.sum(areas)

    # Probability per face
    probs = areas / area_sum

    # Which triangles to sample
    face_idx = np.random.choice(len(faces), size=num_points, p=probs)

    # Selected triangles
    tri_v0 = v0[face_idx]
    tri_v1 = v1[face_idx]
    tri_v2 = v2[face_idx]

    # Random barycentric coordinates
    u = np.sqrt(np.random.rand(num_points, 1))
    v = np.random.rand(num_points, 1)

    # Surface sampling
    samples = (
        (1 - u) * tri_v0 +
        (u * (1 - v)) * tri_v1 +
        (u * v) * tri_v2
    )

    return torch.tensor(samples, dtype=torch.float32)

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
        idx = int(0)
        for v, c in zip(values, counts):
            c = int(c)
            voxels[idx: idx + c] = v
            idx += c

        voxels = voxels.reshape(dims)
        return voxels.astype(np.float32)  # 0/1 float values

class Image2MeshDataset(Dataset):
    def __init__(self, root_dir, split_json, mode="train", transform=None):
        """
        root_dir: path to the R2N2 dataset folder
        split_json: path to split.json
        mode: "train" or "test"
        """
        self.root_dir = root_dir
        self.vox_dir = os.path.join(root_dir, "ShapeNetVoxels")
        self.img_dir = os.path.join(root_dir, "ShapeNetRendering")
        self.transform = transform
        self.images = []    # preloaded image tensors
        self.points = []    # preloaded point cloud tensors
        self.verts = []
        self.faces = []

        # ----------------------------------------------------
        # Load split.json
        # ----------------------------------------------------
        print("Dataset mode: ", mode)
        with open(split_json, "r") as f:
            split = json.load(f)

        assert mode in split, f"{mode} not found in split.json"
        mode_split = split[mode]

        # ----------------------------------------------------
        # Iterate through synset categories
        # Example: "03001627"
        # ----------------------------------------------------
        for synset_id, instances in mode_split.items():
            synset_vox = os.path.join(self.vox_dir, synset_id)
            synset_img = os.path.join(self.img_dir, synset_id)
            

            # ------------------------------------------------
            # For each model instance
            # Example ID: "d8e2e2a923b372731cf97e154cc62f43"
            # ------------------------------------------------
            for model_id, view_ids in instances.items():

                model_vox = os.path.join(synset_vox, model_id)
                model_img = os.path.join(synset_img, model_id)

                # Voxel file
                voxel_path = os.path.join(model_vox, "model.binvox")
                if not os.path.isfile(voxel_path):
                    continue

                voxel_tensor = read_binvox(voxel_path)
                verts, faces = voxel_to_mesh(voxel_tensor)
                point_cloud = sample_points_from_mesh(verts, faces, num_points=2048)
                # ------------------------------------------------
                # Load all views (all 10 views)
                # ------------------------------------------------
                render_dir = os.path.join(model_img, "rendering")
                render_imgs = sorted(glob.glob(os.path.join(render_dir, "*.png")))

                # Make sure we actually have views
                if len(render_imgs) == 0:
                    continue

                # For each image view, preload image + voxel
                for img_path in render_imgs:
                    img_tensor = self._load_image(img_path)

                    self.images.append(img_tensor)
                    self.points.append(point_cloud)
                    self.faces.append(torch.from_numpy(faces.copy()))
                    self.verts.append(torch.from_numpy(verts.copy()))

        print(f"[Dataset Loaded] {mode}: {len(self.images)} image-mesh pairs loaded.")

    # ----------------------
    # Image loader
    # ----------------------
    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    # ----------------------
    # Voxel loader (binvox)
    # ----------------------
    def _load_voxels(self, path):
        import binvox  # if you use a binvox reader library
        with open(path, "rb") as f:
            vox = binvox.read_as_3d_array(f)
        vox = torch.tensor(vox.data).float()  # (32, 32, 32)
        return vox

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "images": self.images[idx],
            "points": self.points[idx],
            "verts": self.verts[idx],
            "faces": self.faces[idx],
        }
    

def load_data(split_path = "./dataset/r2n2_shapenet_dataset/split_03001627.json", r2n2path = "./dataset/r2n2_shapenet_dataset/r2n2", batch_size=32):
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    train_dataset = Image2MeshDataset(r2n2path, split_path, mode = "train", transform=transform)
    test_dataset = Image2MeshDataset(r2n2path, split_path, mode = "test", transform=transform)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print("Example data point keys:", train_dataset[0].keys())
    return train_dataset, test_dataset
if __name__ == "__main__":
    load_data()