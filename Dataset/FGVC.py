from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import os

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
            while line:
                img_name = line.split()[0]
                label = int(line.split()[1])
                self.image_list.append(img_name)
                self.id_list.append(label)
                line = f.readline()
        self.num_classes = max(self.id_list)+1
        
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


def CUB200(scale_size=256, target_size=224):
    path = "./Dataset/CUB_200_2011/images/"
    train_txt = "./Dataset/cub200_train.txt"
    test_txt = "./Dataset/cub200_test.txt"

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

    train_data = CustomDataset(txt_file=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(txt_file=test_txt, root_dir=path, transform=test_transforms)
    num_class = 200

    return train_data, test_data, num_class


def DOG120(scale_size=256, target_size=224):
    path = "./Dataset/Dogs/images/"
    train_txt = "./Dataset/Dogs/Dogs_train.txt"
    test_txt = "./Dataset/Dogs/Dogs_test.txt"

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

    train_data = CustomDataset(txt_file=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(txt_file=test_txt, root_dir=path, transform=test_transforms)
    num_class = 120

    return train_data, test_data, num_class


def CAR196(scale_size=256, target_size=224):
    path = "./Dataset/Cars/"
    train_txt = "./Dataset/Cars/Cars_train.txt"
    test_txt = "./Dataset/Cars/Cars_test.txt"

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

    train_data = CustomDataset(txt_file=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(txt_file=test_txt, root_dir=path, transform=test_transforms)
    num_class = 196

    return train_data, test_data, num_class


def Flowers102(scale_size=256, target_size=224):
    path = "./Dataset/Flowers/102flowers/jpg/"
    train_txt = "./Dataset/Flowers/Flowes_train.txt"
    test_txt = "./Dataset/Flowers/Flowes_test.txt"

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

    train_data = CustomDataset(txt_file=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(txt_file=test_txt, root_dir=path, transform=test_transforms)
    num_class = 102

    return train_data, test_data, num_class


def NABirds555(scale_size=256, target_size=224):
    path = "./Dataset/NABirds/images/"
    train_txt = "./Dataset/NABirds/NABirds_train.txt"
    test_txt = "./Dataset/NABirds/NABirds_test.txt"

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

    train_data = CustomDataset(txt_file=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(txt_file=test_txt, root_dir=path, transform=test_transforms)
    num_class = 555

    return train_data, test_data, num_class
