from itertools import chain
import os
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np
from copy import deepcopy
from PIL import Image

def get_transform(args):
    if 'clip' in args.baseline:
        TRANSFORM = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return TRANSFORM, TRANSFORM
    if 'more' in args.baseline:
        config = {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 0.9}
        TRANSFORM = create_transform(**config)
        return TRANSFORM, TRANSFORM
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std)
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std)
            ]
        )
        return train_transform, test_transform

def get_dataset(args):

    f_name = os.path.join('./sequence', args.sequence_file)

    with open(f_name, 'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()
    dataset_name = random_sep[0]

    if 'C10-5T' in dataset_name:
        DATA_PATH = '/home/haowei/haowei/CL/data/cifar10'
        args.total_class = 10
        args.class_num = int(args.total_class / args.ntasks)
        args.mean = (0.4914, 0.4822, 0.4465)
        args.std = (0.2023, 0.1994, 0.2010)
        train_transform, test_transform = get_transform(args)
        train = datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=train_transform)
        test = datasets.CIFAR10(DATA_PATH, train=False, download=False, transform=test_transform)
    
    elif 'C100-' in dataset_name:
        DATA_PATH = '/home/haowei/haowei/CL/data/cifar100'
        args.total_class = 100
        args.class_num = int(args.total_class / args.ntasks)
        args.mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        args.std = [x / 255 for x in [68.2, 65.4, 70.4]]
        train_transform, test_transform = get_transform(args)
        train = datasets.CIFAR100(DATA_PATH, train=True, download=True, transform=train_transform) 
        test = datasets.CIFAR100(DATA_PATH, train=False, download=False, transform=test_transform)

    elif dataset_name.startswith('T-'):
        args.total_class = 200
        args.class_num = int(args.total_class / args.ntasks)
        train_transform, test_transform = get_transform(args)
        train = datasets.ImageFolder(root='/home/haowei/tiny-imagenet-200/train', transform=train_transform)
        test = datasets.ImageFolder(root='/home/haowei/tiny-imagenet-200/val', transform=test_transform)

    else:
        raise NotImplementedError

    data = {}
    for t in range(args.ntasks):
        data[t] = {}

        cls_id = [int(random_sep[t].split('-')[-1]) * args.class_num + i for i in range(args.class_num)]
        ## train
        train_ = deepcopy(train)
        targets_aux, data_aux, full_target_aux, names_aux = [], [], [], []
        idx_aux = []

        for c in cls_id:
            idx = np.where(np.array(train.targets) == c)[0]
            
            if dataset_name.startswith('T-'):   # for tinyImagenet
                idx_aux.append(idx)
            else:
                data_aux.append(train.data[idx])
                targets_aux.append(np.zeros(len(idx), dtype=np.int) + c)
                full_target_aux.append([[c, c]] for _ in range(len(idx)))
                names_aux.append([str(c) for _ in range(len(idx))])
        
        # mix replay dataset
        if args.training:
            cls_id_past = []
            for t_ in range(t):
                cls_id_past = cls_id_past + [int(random_sep[t_].split('-')[-1]) * args.class_num + i for i in range(args.class_num)]
            for c in cls_id_past:
                idx = np.where(np.array(train.targets) == c)[0][:args.replay_buffer_size // cls_id[0]]
                if dataset_name.startswith('T-'):
                    idx_aux.append(idx)
                else:
                    data_aux.append(train.data[idx])
                    targets_aux.append(np.zeros(len(idx), dtype=np.int) + c)
                    full_target_aux.append([[c, c]] for _ in range(len(idx)))
                    names_aux.append([str(c) for _ in range(len(idx))])
        
        if dataset_name.startswith('T-'):
            idx_list = np.concatenate(idx_aux)
            train_ = Subset(train_, idx_list)
            train_.data = []
            train_.targets = np.array(train_.dataset.targets)[idx_list]
            train_.transform = train_.dataset.transform
        else:
            train_.data = np.array(list(chain(*data_aux)))
            train_.targets = np.array(list(chain(*targets_aux)))
            train_.full_labels = np.array(list(chain(*full_target_aux)))
            train_.names = list(chain(*names_aux))
        del data_aux, targets_aux, full_target_aux, names_aux, idx_aux
        data[t]['train'] = train_

        ## test
        test_ = deepcopy(test)
        targets_aux, data_aux, full_target_aux, names_aux = [], [], [], []
        idx_aux = []
        for c in cls_id:
            idx = np.where(np.array(test.targets) == c)[0]
            if dataset_name.startswith('T-'):
                idx_aux.append(idx)
            else:
                data_aux.append(test.data[idx])
                targets_aux.append(np.zeros(len(idx), dtype=np.int) + c)
                full_target_aux.append([[c, c]] for _ in range(len(idx)))
                names_aux.append([str(c) for _ in range(len(idx))])
        
        if dataset_name.startswith('T-'):
            idx_list = np.concatenate(idx_aux)
            test_ = Subset(test_, idx_list)
            test_.data = []
            test_.targets = np.array(test_.dataset.targets)[idx_list]
            test_.transform = test_.dataset.transform
        else:
            test_.data = np.array(list(chain(*data_aux)))
            test_.targets = np.array(list(chain(*targets_aux)))
            test_.full_labels = np.array(list(chain(*full_target_aux)))
            test_.names = list(chain(*names_aux))
        del data_aux, targets_aux, full_target_aux, names_aux
        data[t]['test'] = test_

    return data
