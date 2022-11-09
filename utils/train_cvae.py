import torch

def weighted_average(tensor, weights=None, dim=0):
    """Computes weighted average of [tensor] over dimension [dim]."""

    if weights is None:
        mean = torch.mean(tensor, dim=dim)
    else:
        batch_size = tensor.size(dim) if len(tensor.size()) > 0 else 1
        assert len(weights) == batch_size
        norm_weights = torch.tensor([weight for weight in weights]).to(tensor.device)
        mean = torch.mean(norm_weights * tensor, dim=dim)
    return mean

def train_cvae(model, train_loader, device, params_dict_name, dataset='cifar10'):
    """Trains CVAE on the given dataset."""

    if dataset == 'mnist':
        n_epochs = 100
    elif dataset == 'cifar10':
        n_epochs = 200
    else:
        raise ValueError
    LR = 0.001

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    for epoch in range(n_epochs):
        for data, y in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            y = y.long()
            y_onehot = torch.Tensor(y.shape[0], model.class_num)
            y_onehot.zero_()
            y_onehot.scatter_(1, y.view(-1, 1), 1)
            y_onehot = y_onehot.to(device)
            mu, logvar, recon = model(data, y_onehot)

            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            variatL = weighted_average(variatL, weights=None, dim=0)
            variatL /= (model.image_channels * model.image_size * model.image_size)

            data_resize = data.reshape(-1, model.image_channels * model.image_size * model.image_size)
            recon_resize = recon.reshape(-1, model.image_channels * model.image_size * model.image_size)
            reconL = (data_resize - recon_resize) ** 2
            reconL = torch.mean(reconL, 1)
            reconL = weighted_average(reconL, weights=None, dim=0)

            loss = variatL + reconL

            loss.backward()
            optimizer.step()

        print("epoch: {}, loss = {}, reconL = {}, variaL = {}".format(epoch, loss, reconL, variatL))

    torch.save(model.state_dict(), params_dict_name)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    import sys
    sys.path.append('/home/haowei/haowei/CL')
    from config import parse_args
    from networks.baseline.VAE import ConditionalVAE
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import os
    args = parse_args()
    setup_seed(args.seed)
    args.class_num = 2
    DATA_PATH = os.path.join(args.base_dir, 'cifar10')
    from dataloader import data
    train_dataset = data.get_subclass_dataset(
        datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=transforms.ToTensor(), target_transform=lambda x: x - args.class_num * args.task),
        [0,1]
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    model = ConditionalVAE(image_channels=3, image_size=32, dataset='cifar10').to('cuda')

    train_cvae(model, train_loader, 'cuda', os.path.join(args.base_dir, 'cifar10_CVAE/{}.pt'.format(args.task)), dataset='cifar10')