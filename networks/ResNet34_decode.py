import torch
from torch import nn
from .resnet import resnet34,resnet50,resnet9,resnet18


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv2d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm2d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm2d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class C4_head(nn.Module):
    def __init__(self,in_channel=256,out_channel=512):
        super(C4_head, self).__init__()

        self.conv1 = nn.Conv2d(in_channel,out_channel, kernel_size=(3,3,3), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3, 2), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channel, out_channel*2, kernel_size=(2,2,1), stride=1, padding=0, bias=False)

    def forward(self, x, bs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        bs_num,c,w,h,d = x.shape
        x= torch.reshape(x,(bs,bs_num//bs,c*w*h*d))
        return x

class C5_head(nn.Module):
    def __init__(self,in_channel=512,out_channel=1024):
        super(C5_head, self).__init__()


        self.conv1 = nn.Conv2d(in_channel,out_channel, kernel_size=(3,3,2), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, bs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        bs_num, c, w, h, d = x.shape
        x = torch.reshape(x, (bs, bs_num // bs, c * w * h * d))
        return x

class Resnet34(nn.Module):
    def __init__(self, resnet_encoder=None, n_channels=3, n_classes=2, n_filters=24, normalization='batchnorm', has_dropout=True):
        super(Resnet34, self).__init__()
        self.has_dropout = has_dropout
        # self.resnet_encoder = resnet34()
        self.resnet_encoder = resnet18()

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(2, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(2, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(2, n_filters, n_filters, normalization=normalization)
        # self.block_nine_up = UpsamplingDeconvBlock(n_filters , n_filters, normalization=normalization)

        self.out_conv = nn.Conv2d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout2d(p=0.5, inplace=False)
        self.memory = nn.Parameter(torch.zeros(1,192,28,28),requires_grad=True)
        # self.memory = nn.Parameter(torch.zeros(1,384,14,14),requires_grad=True)

        self.gate1 = gate_xb(192)
        self.gate2 = gate_xb(96)
        self.gate3 = gate_xb(48)
        self.gate4 = gate_xb(24)

        self.__init_weight()


    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        B,_,_,_ = x1.shape
        memory = self.memory.repeat(B,1,1,1)
        # memory = x5
        x5_up = self.block_five_up(x5)
        x5_up,memory = self.gate1(x5_up,memory,x4)

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up,memory = self.gate2(x6_up,memory,x3)

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up,memory = self.gate3(x7_up,memory,x2)

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up,memory = self.gate4(x8_up,memory,x1)

        x9 = self.block_nine(x8_up)
        # x9 = self.block_nine_up(x9)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out

    def forward(self, input):
        resnet_features = self.resnet_encoder(input)
        out = self.decoder(resnet_features)
        return out

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class xb(nn.Module):
    def __init__(self,x2dim):
        super(xb, self).__init__()
        if x2dim == 192:
            self.conv = nn.Conv2d((x2dim*3)//2,x2dim * 2,kernel_size=3,stride=1,padding=1)
        else:
            self.conv = nn.Conv2d(x2dim*2,x2dim * 2,kernel_size=3,stride=1,padding=1)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x1, x2):
        B,C,H,W = x2.shape
        if C ==192:
            x_in = torch.cat((x1.reshape(B,C//2,H,W),x2),dim=1)
        else:
            x_in = torch.cat((x1,x2),dim=1)
        x_out = self.conv(x_in)
        out = self.act(x_out)
        out = out.reshape(B,C//2,H * 2,W * 2)
        return out


class gate(nn.Module):
    def __init__(self,dim):
        super(gate, self).__init__()
        self.up = UpsamplingDeconvBlock(dim * 2, dim, normalization='none')
        self.linear = nn.Linear(dim,dim*4)
        self.sig = nn.Sigmoid()
        self.act = nn.Tanh()
    def forward(self, x_in, memory):
        memory = self.up(memory)
        x1234 = self.linear(x_in.permute(0,2,3,1)).permute(0,3,1,2)
        x1,x2,x3,x4 = torch.chunk(x1234,chunks=4,dim=1)
        x1 = self.sig(x1)
        memory = x1 * memory + self.sig(x2)*self.act(x3)
        x_out = memory * self.sig(x4)
        return x_out,memory

class gate2(nn.Module):
    def __init__(self,dim,p=0.0):
        super(gate2, self).__init__()
        # self.up = UpsamplingDeconvBlock(dim * 2, dim, normalization='none')
        self.linear_up = nn.Linear(dim * 2,dim * 4)
        self.linear = nn.Linear(dim*2,dim*4)
        self.sig = nn.Sigmoid()
        self.act = nn.Tanh()
        self.drop = nn.Dropout(p)
        # self.sig = nn.Softmax()
        # self.act = nn.ReLU()
    def forward(self, x_in, memory,skip):
        # memory = self.up(memory)
        B,C,H,W = x_in.shape
        _,C_memory,_,_ = memory.shape
        if C != C_memory:
            memory = self.linear_up(memory.permute(0,2,3,1)).permute(0,3,1,2)
            memory = memory.reshape(B,C,H,W)
        x1234 = self.linear(torch.cat((x_in.permute(0,2,3,1),skip.permute(0,2,3,1)),dim=3)).permute(0,3,1,2)
        x1,x2,x3,x4 = torch.chunk(x1234,chunks=4,dim=1)
        x1 = self.sig(x1)
        memory = x1 * memory + self.sig(x2)*self.act(x3)
        x_out = self.act(memory) * self.sig(x4)
        if C != C_memory:
            x_out = self.drop(x_out)
            memory = self.drop(memory)
        return x_out,memory
class gate_xb(nn.Module):
    def __init__(self,dim,p=0.0):
        super(gate_xb, self).__init__()
        self.up = UpsamplingDeconvBlock(dim * 2, dim, normalization='batchnorm')
        # self.linear_up = nn.Linear(dim * 2,dim * 4)
        self.xh_conv3 = nn.Conv2d(in_channels=dim*2,out_channels=dim*3,kernel_size=3,padding=1)
        self.xc_conv3 = nn.Conv2d(in_channels=dim*2,out_channels=dim*2,kernel_size=3,padding=1)

        # self.xh_conv1 = nn.Conv2d(in_channels=dim*2,out_channels=dim*3,kernel_size=1)
        # self.xh_dconv = nn.Conv2d(in_channels=dim*3,out_channels=dim*3,kernel_size=3,padding=1,groups=dim*3)

        # self.xc_conv1 = nn.Conv2d(in_channels=dim*2,out_channels=dim*2,kernel_size=1)
        # self.xc_dconv = nn.Conv2d(in_channels=dim*2,out_channels=dim*2,kernel_size=3,padding=1,groups=dim*2)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(p)
        # self.sig = nn.Softmax()
        # self.act = nn.ReLU()
    def forward(self, x_in, memory,skip):
        B,C,H,W = x_in.shape
        _,C_memory,_,_ = memory.shape
        # if C != C_memory:
        #     memory = self.up(memory)
            # memory = memory.reshape(B,C,H,W)

        # xh = torch.cat((x_in,skip),dim=1)
        # xc = torch.cat((x_in,memory),dim=1)

        # xh_fic = self.xh_conv3(xh)
        # xc_ic = self.xc_conv3(xc)


        # xh_f,xh_i,xh_c = torch.chunk(xh_fic,dim=1,chunks=3)
        # memory = memory * self.sig(xh_f) + self.sig(xh_i) * xh_c

        # xc_i,xc_c = torch.chunk(xc_ic,dim=1,chunks=2)
        # x_out = x_in + self.sig(xc_i) * xc_c
        x_out = x_in + skip
        return x_out,memory