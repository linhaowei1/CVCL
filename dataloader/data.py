import os
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

def get_subclass_dataset(dataset, classes, size=None):
    if not isinstance(classes, list):
        classes = [classes]
    indices = []
    if size is not None:
        size_cnt = {cls:0 for cls in classes}
        for idx, data in enumerate(dataset):
            if data[1] in classes:
                if size_cnt[data[1]] == size: continue
                else:
                    size_cnt[data[1]] += 1
                indices.append(idx)
    else:
        for idx, data in enumerate(dataset):
            if data[1] in classes:
                indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset

def get_transform(args):
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
    """Get datasets for setting 1 (OOD Detection on the Same Dataset)."""

    f_name = os.path.join('./sequence', args.sequence_file)
    data = {}

    with open(f_name, 'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    for t in range(args.task + 1):
        dataset_name = random_sep[t]
        data[t] = {}
        data[t]['name'] = dataset_name

        print('dataset_name: ', dataset_name)

        if 'C10-5T' in dataset_name:
            DATA_PATH = os.path.join(args.base_dir, 'cifar10')
            task_id = int(dataset_name.split('-')[-1])
            args.total_class = 10
            args.class_num = int(args.total_class / args.ntasks)
            args.mean = (0.4914, 0.4822, 0.4465)
            args.std = (0.2023, 0.1994, 0.2010)
            train_transform, test_transform = get_transform(args)
            data[t]['train'] = get_subclass_dataset(
                datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=train_transform), 
                [args.class_num * task_id + i for i in range(args.class_num)]
            )
            data[t]['test'] = get_subclass_dataset(
                datasets.CIFAR10(DATA_PATH, train=False, download=False, transform=test_transform), 
                [args.class_num * task_id + i for i in range(args.class_num)]
            )
            data[t]['replay'] = get_subclass_dataset(
                datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=train_transform), 
                [args.class_num * task_id + i for i in range(args.class_num)],
                size = args.replay_buffer_size
            )

        
        elif 'C100-' in dataset_name:
            DATA_PATH = os.path.join(args.base_dir, 'cifar100')
            task_id = int(dataset_name.split('-')[-1])
            args.total_class = 100
            args.class_num = int(args.total_class / args.ntasks)
            args.mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            args.std = [x / 255 for x in [68.2, 65.4, 70.4]]
            train_transform, test_transform = get_transform(args)
            data[t]['train'] = get_subclass_dataset(
                datasets.CIFAR100(DATA_PATH, train=True, download=True, transform=train_transform), 
                [args.class_num * task_id + i for i in range(args.class_num)]
            )
            data[t]['test'] = get_subclass_dataset(
                datasets.CIFAR100(DATA_PATH, train=False, download=False, transform=test_transform), 
                [args.class_num * task_id + i for i in range(args.class_num)]
            )

        else:
            raise NotImplementedError

    return data
