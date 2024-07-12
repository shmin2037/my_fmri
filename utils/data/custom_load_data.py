# custom_load_data.py
import h5py
from torch.utils.data import Dataset, DataLoader
from utils.data.custom_transforms import data_transform
from pathlib import Path

class SliceData(Dataset):
    def __init__(self, root, transform, input_key="image_input", target_key="image_label"):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.image_examples = []

        image_files = list(Path(root / "image").iterdir())
        for fname in sorted(image_files):
            num_slices = self._get_metadata(fname)

            self.image_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
            else:
                raise KeyError(f"Neither {self.input_key} nor {self.target_key} found in {fname}")
        return num_slices

    def __len__(self):
        return len(self.image_examples)

    def __getitem__(self, i):
        image_fname, dataslice = self.image_examples[i]

        with h5py.File(image_fname, "r") as hf:
            input_data = hf[self.input_key][dataslice]
            target = hf[self.target_key][dataslice]

        return data_transform(input_data, target, image_fname.name, dataslice)



def create_data_loader(data_path, batch_size, shuffle=True, transform=None):
    dataset = SliceData(root=data_path, transform=data_transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
