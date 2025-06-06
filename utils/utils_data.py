import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import json
import random
from tqdm import tqdm
from joblib import dump, load
from torch.utils import data
import pandas as pd
from torchvision import transforms as T

class Preprocessing_Layer(torch.nn.Module):
    def __init__(self, mean, std):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std

    def preprocess(self, img, mean, std):
        img = img.clone()
        #img /= 255.0

        img[:,0,:,:] = (img[:,0,:,:] - mean[0]) / std[0]
        img[:,1,:,:] = (img[:,1,:,:] - mean[1]) / std[1]
        img[:,2,:,:] = (img[:,2,:,:] - mean[2]) / std[2]

        #img = img.transpose(1, 3).transpose(2, 3)
        return(img)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        res = self.preprocess(x, self.mean, self.std)
        return res

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ImagePathDataset(VisionDataset):
    def __init__(self, config, transform=None, target_transform=None,
                 loader=default_loader, return_paths=False, sample_num=5000):
        super().__init__(root=config["root"], transform=transform, target_transform=target_transform)
        self.config = config

        self.loader = loader
        self.extensions = IMG_EXTENSIONS

        self.classes = config["classes"]
        self.class_to_idx = config["class_to_idx"]
        self.samples = random.sample(config["samples"], sample_num)
        self.targets = [s[1] for s in self.samples]
        self.return_paths = return_paths

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        output = sample, target

        if self.return_paths:
            return output, path
        else:
            return output

    def __len__(self):
        return len(self.samples)

    @classmethod
    def from_path(cls, config_path, sample_num=5000, *args, **kwargs):
        with open(config_path, mode="r") as f:
            return cls(config=json.loads(f.read()), sample_num=sample_num, *args, **kwargs)

    
def set_requires_grad(named_parameters, requires_grad):
    for name, param in named_parameters:
        param.requires_grad = requires_grad



def save_images(images, img_list, idx, output_dir):
    """Saves images to the output directory.
        Args:
          images: tensor with minibatch of images
          img_list: list of filenames 
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
          output_dir: directory where to save images
    """
    for i in range(idx):
        filename = os.path.basename(img_list[i])
        cur_images = (images[i, :, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)

        im = Image.fromarray(cur_images)
        im.save('{}.png'.format(os.path.join(output_dir, filename[:-4])))

def save_images_pt(images, img_list, idx, output_dir):
    """Saves images as PyTorch .pt files to the output directory.
    
    Args:
      images: tensor with minibatch of images
      img_list: list of filenames 
        If number of filenames in this list is less than the number of images in
        the minibatch, then only the first len(filenames) images will be saved.
      idx: number of images to save
      output_dir: directory where to save images
    """
    for i in range(idx):
        # Extract the base filename without extension
        filename = os.path.basename(img_list[i])
        base_filename = filename[:-4]  # Remove the file extension

        # Prepare the tensor to save
        cur_image_tensor = images[i]

        # Define the output path
        output_path = os.path.join(output_dir, f"{base_filename}.pt")

        # Save the tensor as a .pt file
        torch.save(cur_image_tensor, output_path)


def save_image(images, index, output_dir, file_name):
    cur_image = (images[index, :, :, :].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    im = Image.fromarray(cur_image)
    im.save('{}.png'.format(os.path.join(output_dir, file_name)))

class ImageNet(data.Dataset):
    def __init__(self, dir, csv_path, transforms = None, sample_num=64):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms
        self.sample_num = sample_num
        self.sign = False

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        if not self.sign:
            ImageID = img_obj['ImageId'] + '.png'
        else:
            ImageID = img_obj['ImageId'][:-1] + '.png'

        Truelabel = img_obj['TrueLabel'] - 1
        TargetClass = img_obj['TargetClass'] - 1
        try:
            img_path = os.path.join(self.dir, ImageID)
            pil_img = Image.open(img_path).convert('RGB')
        except:
            self.sign = True
            ImageID = img_obj['ImageId'][:-1] + '.png'
            img_path = os.path.join(self.dir, ImageID)
            pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        return (data, Truelabel, TargetClass), ImageID

    def __len__(self):
        return self.sample_num