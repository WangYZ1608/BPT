from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from timm.data import create_transform
import matplotlib.pyplot as plt
import os, torch

class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, training=False):
        self.image_list = []
        self.id_list = []
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = 0
        self.training = training
        with open(txt_file, 'r') as f:
            line = f.readline()
            # self.datas = f.readlines()
            while line:
                img_name = line.split()[0]
                label = int(line.split()[1])
                # label = int(label)
                self.image_list.append(img_name)
                self.id_list.append(label)
                line = f.readline()
        self.num_classes = max(self.id_list)+1
        assert self.num_classes == 1000
        
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label = self.id_list[idx]
        img_name = os.path.join(self.root_dir,img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image,label


def ImageNet(args=None, scale_size=256, target_size=224):
    train_path = "./DataSet/ImageNet2012/ILSVRC2012_img_train/"
    train_txt = "./Dataset/IN1K_train.txt"
    test_path = "./DataSet/ImageNet2012/ILSVRC2012_img_val/"
    test_txt = "./Dataset/IN1K_val.txt"

    train_transforms = T.Compose([
        T.RandomResizedCrop(target_size, interpolation=3),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(scale_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(txt_file=train_txt, root_dir=train_path, transform=train_transforms)
    test_data = CustomDataset(txt_file=test_txt, root_dir=test_path, transform=test_transforms)
    num_class = 1000

    return train_data, test_data, num_class