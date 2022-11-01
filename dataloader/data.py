import os
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]
    indices = []
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
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    return train_transform, test_transform

def get_dataset(args):
    """Get datasets for setting 1 (OOD Detection on the Same Dataset)."""

    f_name = os.path.join('./sequence', args.sequence_file)
    data = {}

    with open(f_name, 'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    args.ntasks = len(random_sep)
    for t in range(args.task + 1):
        dataset_name = random_sep[t]
        data[t] = {}
        data[t]['name'] = dataset_name

        print('dataset_name: ', dataset_name)

        if 'C10' in dataset_name:
            DATA_PATH = os.path.join(args.base_dir, 'cifar10')
            task_id = int(dataset_name.split('-')[-1])
            args.total_class = 10
            args.class_num = int(args.total_class / args.ntasks)
            train_transform, test_transform = get_transform(args)
            data[t]['train'] = get_subclass_dataset(
                datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=train_transform), 
                [args.class_num * task_id + i for i in range(args.class_num)]
            )
            data[t]['test'] = get_subclass_dataset(
                datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=test_transform), 
                [args.class_num * task_id + i for i in range(args.class_num)]
            )
        else:
            raise NotImplementedError

    return data
