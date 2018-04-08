import time
import numpy as np
from skimage import io, transform, img_as_float
from pathlib import Path

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate


def load_volume_and_labels(list_name, target_size=(128, 128)):
    """Load video volume via list_name, which contains multiple lines of <path, label> pair
        `/path/to/image/KLAC0421_043.jpg 8`
        `/path/to/image/KLAC0421_044.jpg 2`
        `/path/to/image/KLAC0421_045.jpg 0`
        ...
    Args:
        list_name: name of certain video's frame list
        target_size: resize single frame to fixed size, default is (112, 112)
    Return:
        volume: <float64 3D-array> [length, height, width] in range [0, 1]
        labels: <float64 1D-array> [length]
    """
    with open(list_name) as f:
        lines = f.readlines()
    lines = [l.strip().split() for l in lines]
    volume = []
    labels = []
    for image_path, label in lines:
        image = img_as_float(io.imread(image_path, as_grey=True))
        resized = transform.resize(image, target_size, mode='reflect')
        volume.append(resized)
        labels.append(float(label))
    volume = np.array(volume)
    labels = np.array(labels)
    return volume, labels


class PanelDataset(data.Dataset):
    def __init__(self, list_root='list', stage='train', fixed_length=64, compress=100):
        """Assume texts of <image_path, label> is in directory 'list/train' or 'list/test'
        Args:
            fixed_length: sample the fixed length of videos for convenient upsample
            compress: compress Dataloader size, for faster epoch
            1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
            2. Preprocess the data (e.g. torchvision.Transform).
            3. Return a data pair (e.g. volume and label).
        """
        super(PanelDataset, self).__init__()
        self.list_root = Path(list_root)
        self.stage = stage
        self.fixed_length = fixed_length
        self.compress = compress
        self.video_paths = list((self.list_root / stage).iterdir())

    def __getitem__(self, index):
        # Set random seed for random augment.
        np.random.seed(int(time.time()))
        index += np.random.choice(self.compress, 1)[0] * (len(self.video_paths) // self.compress)

        # Load volume and label via list
        volume, label = load_volume_and_labels(self.video_paths[index])

        # Slice the fixed length
        sample_range = volume.shape[0] - self.fixed_length
        while sample_range < 0:
            volume = np.concatenate([volume, volume])
            label = np.concatenate([label, label])
            sample_range = volume.shape[0] - self.fixed_length

        if sample_range == 0:
            start_i = 0
        else:
            start_i = np.random.randint(low=0, high=sample_range)
        selected = slice(start_i, start_i + self.fixed_length)
        volume, label = volume[selected], label[selected]

        # Currently without augmentation
        # Trans to tensor [Channel(1), Depth, Height, Width]
        volume = torch.from_numpy(np.expand_dims(volume, 0))  # expand gray channel axis
        label = torch.from_numpy(label)
        return volume, label

    def __len__(self):
        return len(self.video_paths) // self.compress


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    return default_collate(batch)


if __name__ == '__main__':
    import visdom
    import torchvision

    vis = visdom.Visdom(env='inputs')

    dataset = PanelDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for volume, labels in dataloader:
        print(volume.size())
        print(labels[0][:36])
        images = volume[0].permute(1, 0, 2, 3)[:36]

        grid = torchvision.utils.make_grid(images, 6)
        vis.image(grid)
        torchvision.utils.save_image(grid, 'debug/check_input.jpg')
        break
