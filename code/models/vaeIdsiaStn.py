# This code is modified from the repository
# https://github.com/bhpfelix/Variational-Autoencoder-PyTorch

import torch.nn as nn
import torch
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
   
    def forward(self, x):
        numel = x.numel() / x.shape[0]
        return x.view(-1, int(numel)) 

def convNoutput(convs, input_size, input_channels):
    output = torch.zeros(1, input_channels, int(input_size))
    with torch.no_grad():
        for conv in convs:
            output = conv(output)
    return output.numel(), output.shape

class stn(nn.Module):
    def __init__(self, input_channels, input_size, params):
        super(stn, self).__init__()

        input_size = int(input_size)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
                    nn.Conv1d(input_channels, params[0], kernel_size=5, stride=1, padding=2),
                    nn.ReLU(True),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv1d(params[0], params[1], kernel_size=5, stride=1, padding=2),
                    nn.ReLU(True),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                    )

        out_numel, _ = convNoutput(
            [self.maxpool, self.conv1, self.conv2], input_size, input_channels)
        # set fc layer based on predicted size
        self.fc = nn.Sequential(
                View(),
                nn.Linear(out_numel, params[2]),
                nn.ReLU()
                )
        self.classifier = classifier = nn.Sequential(
                View(),
                nn.Linear(params[2], 2)
                )
        # Initialize the 1D affine transform to identity (unit scale, no shift).
        self.classifier[1].weight.data.fill_(0)
        self.classifier[1].bias.data = torch.FloatTensor([1, 0])

    def localization_network(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x


    def forward(self, x):
        affine = self.localization_network(x)
        zero = torch.zeros_like(affine[:, 0])
        one = torch.ones_like(affine[:, 0])
        theta = torch.stack(
            (affine[:, 0], zero, affine[:, 1], zero, one, zero), dim=1
        ).view(-1, 2, 3)
        signal = x.unsqueeze(2)
        grid = F.affine_grid(theta, signal.size())
        return F.grid_sample(signal, grid).squeeze(2)


class VAEIdsia(nn.Module):
    def __init__(self, nc, input_size, latent_variable_size=300, cnn_chn=(100, 150, 250), param1=None, param2=None, param3=None):
        super(VAEIdsia, self).__init__()

        self.cnn_chn = cnn_chn
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

        self.input_size = int(input_size)
        self.nc = nc
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv1d(nc, self.cnn_chn[0], 7, 2, 3) # inchn, outchn, kernel, stride, padding, dilation, groups
        self.bn1 = nn.BatchNorm1d(self.cnn_chn[0])

        self.e2 = nn.Conv1d(self.cnn_chn[0], self.cnn_chn[1], 4, 2, 1) # 1/4
        self.bn2 = nn.BatchNorm1d(self.cnn_chn[1])

        self.e3 = nn.Conv1d(self.cnn_chn[1], self.cnn_chn[2], 4, 2, 1) # 1/8
        self.bn3 = nn.BatchNorm1d(self.cnn_chn[2])

        encoded_numel, encoded_shape = convNoutput(
            [self.e1, self.e2, self.e3], self.input_size, nc)
        self.encoded_size = encoded_shape[-1]
        self.fc1 = nn.Linear(encoded_numel, latent_variable_size)
        self.fc2 = nn.Linear(encoded_numel, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, encoded_numel)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest') # 8 -> 16
        self.pd1 = nn.ReplicationPad1d(1)
        self.d2 = nn.Conv1d(self.cnn_chn[2], self.cnn_chn[1], 3, 1)
        self.bn6 = nn.BatchNorm1d(self.cnn_chn[1], 1.e-3)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest') # 16 -> 32
        self.pd2 = nn.ReplicationPad1d(1)
        self.d3 = nn.Conv1d(self.cnn_chn[1], self.cnn_chn[0], 3, 1)
        self.bn7 = nn.BatchNorm1d(self.cnn_chn[0], 1.e-3)

        self.up3 = nn.Upsample(size=self.input_size, mode='nearest')
        self.pd3 = nn.ReplicationPad1d(1)
        self.d4 = nn.Conv1d(self.cnn_chn[0], nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if param1 is not None:
            self.stn1 = stn(nc, self.input_size, param1)
        if param2 is not None:
            _, h1_shape = convNoutput([self.e1], self.input_size, nc)
            self.stn2 = stn(self.cnn_chn[0], h1_shape[-1], param2)
        if param3 is not None:
            _, h2_shape = convNoutput(
                [self.e1, self.e2], self.input_size, nc)
            self.stn3 = stn(self.cnn_chn[1], h2_shape[-1], param3)


    def encode(self, x):
        if self.param1 is not None:
            x = self.stn1(x)

        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        if self.param2 is not None:
            h1 = self.stn2(h1)

        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        if self.param3 is not None:
            h2 = self.stn3(h2)

        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = h3.view(h3.size(0), -1)

        return self.fc1(h4), self.fc2(h4), x

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.cnn_chn[2], self.encoded_size)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        return self.sigmoid(self.d4(self.pd3(self.up3(h3))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar, xstn = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar, xstn

    def init_params(self, net):
        print('Loading the model from the file...')
        net_dict = self.state_dict()
        if isinstance(net, dict):
            pre_dict = net
        else:
            pre_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pre_dict = {k: v for k, v in pre_dict.items() if (k in net_dict)} # for fs net
        net_dict.update(pre_dict)
        # 3. load the new state dict
        self.load_state_dict(net_dict)