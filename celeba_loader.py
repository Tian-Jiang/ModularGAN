import os
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)


    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        image = self.transform(image)
        return image, torch.FloatTensor(label)
        #return image.to(torch.float16), torch.HalfTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def data_loader(image_dir, attr_path, selected_attr, crop_size=178, image_size=128, batchsz=16, mode='train'):
    celeba_train = CelebA(image_dir, attr_path, selected_attr,
                          transform=transforms.Compose([transforms.CenterCrop(crop_size),
                                                        transforms.Resize(image_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                          mode=mode)

    celeba_data = data.DataLoader(dataset=celeba_train, batch_size=batchsz, shuffle=(mode=='train'))

    return celeba_data
