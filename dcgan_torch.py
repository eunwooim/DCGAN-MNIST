import random

import torch
from torch.cuda import is_available
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms

def trainset(params):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    dataset = torchvision.datasets.MNIST(
        root=params['root'],train=True,download=True,transform=transform
        )
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=params['batch_size'],shuffle=True, 
        num_workers=params['workers']
        )
    cuda_avble = torch.cuda.is_available()
    device = torch.device('cuda' if params['cuda'] and cuda_avble else 'cpu')

    return trainloader, device

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        # self.ngpu = ngpu
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(params['nz'], params['ngf'] * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(params['ngf'] * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(params['ngf'] * 8, params['ngf'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ngf'] * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( params['ngf'] * 4, params['ngf'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ngf'] * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( params['ngf'] * 2, params['ngf'], 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ngf']),
            nn.ReLU(True),
            nn.ConvTranspose2d( params['ngf'], params['nc'], 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(params['nc'], params['ndf'], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(params['ndf'], params['ndf'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ndf'] * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(params['ndf'] * 2, params['ndf'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ndf'] * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(params['ndf'] * 4, params['ndf'] * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ndf'] * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(params['ndf'] * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

if __name__ == '__main__':
    params = {
    'root' : 'C:\Users\eunwoo\Desktop\Code\savehere',
    'cuda' : True,
    'batch_size' : 128,
    'workers' : 2,
    'imsize' : 28,
    'nc' : 1,
    'nz' : 100,
    'ngf' : 64,
    'ndf' : 64,
    'epochs' : 10,
    'lr' : 2e-4,
    'beta1' : 0.5,
    'save_epoch' : 2,
    'visualize' : 16,
    'seed' : 999}

    random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    trainloader, device = trainset(params)
    generator = Generator().to(device)
    generator.apply(weights_init)
    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    noise = torch.randn(params['visualize'],params['nz'],1,1,device=device)

    real_label = 1
    fake_label = 0

    optimizerD = optim.Adam(discriminator.parameters(),
                            lr=params['lr'], betas=(params['beta1'], 0.999))
    optimizerG = optim.Adam(generator.parameters(),
                            lr=params['lr'], betas=(params['beta1'], 0.999))

    img_list = []
    G_losses = []
    D_losses = []

    for epoch in range(params['epochs']):
        for i, data in enumerate(trainloader):
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,),real_label,device=device)
            output = discriminator(real_data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = discriminator(noise)
            label.fill_(fake_label)

            output = discriminator(fake_data.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()
            
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake_data).view(-1)
            errG = criterion(output, label)
            errG.backward()

            D_G_z2 = output.mean().item()
            optimizerG.step()

            print(torch.cuda.is_available())
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, params['nepochs'], i, len(trainloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            D_losses.append(errD.item())
