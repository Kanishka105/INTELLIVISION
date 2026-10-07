import cv2

from pathlib import Path

from torch.utils.data import Dataset, DataLoader


class VehicleDataset(Dataset):
    """
    Dataset class for loading vehicle images.
    """

    def __init__(self, image_dir):

        self.image_dir = Path(image_dir)

        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        }

        self.image_paths = sorted([
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in valid_extensions
        ])

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {self.image_dir}"
            )

        print(
            f"Found {len(self.image_paths)} images"
        )

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise ValueError(
                f"Could not read {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        return image, str(image_path)


def create_dataloader(
    image_dir,
    batch_size=8,
    shuffle=True,
    num_workers=0
):
    """
    Create PyTorch DataLoader.
    """

    dataset = VehicleDataset(
        image_dir
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return loader