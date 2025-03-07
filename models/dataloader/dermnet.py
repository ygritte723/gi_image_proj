import numpy as np
import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Dermnet(Dataset):
    def __init__(self, setname, args, return_path=False):
        IMAGE_PATH = os.path.join(args.data_dir, "dermnet/images")
        SPLIT_PATH = os.path.join(args.data_dir, "dermnet/split_30")

        csv_path = osp.join(SPLIT_PATH, setname + ".csv")
        lines = [x.strip() for x in open(csv_path, "r").readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []
        self.args = args
        self.key = {}

        for l in lines:
            # print(l)
            name, wnid = l.split(",")
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                self.key[wnid] = lb
            data.append(path)
            label.append(self.key[wnid])

        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.num_class = len(set(label))
        # print(data)
        # print(label)
        # print(self.num_class)
        # print(setname)
        self.return_path = return_path

        if setname == "val" or setname == "test":
            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose(
                [
                    transforms.Resize([resize_size, resize_size]),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    # cifar10
                    transforms.Normalize(
                        np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                        np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]),
                    ),
                ]
            )
        elif setname == "train":
            image_size = 84
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                        np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]),
                    ),
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert("RGB"))
        # print(image,label,path)
        if self.return_path:
            return image, label, path
        else:
            return image, label


if __name__ == "__main__":
    pass
