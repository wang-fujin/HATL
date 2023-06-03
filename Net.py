import torch
import torch.nn as nn
from DSBN import DomainSpecificBatchNorm1D

class ChannelAttention(nn.Module):
    # 每个通道返回一个权值
    def __init__(self,input_channel,ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveAvgPool1d(1)

        self.layer1 = nn.Conv1d(input_channel, input_channel//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Conv1d(input_channel // ratio, input_channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = self.layer2(self.relu1(self.layer1(self.avg_pool(x))))
        max_out = self.layer2(self.relu1(self.layer1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class ResNet1dBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride, use_attention=True, output_attention=True):
        super(ResNet1dBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = DomainSpecificBatchNorm1D(output_channel)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = DomainSpecificBatchNorm1D(output_channel)

        self.ca = ChannelAttention(output_channel)

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.output_attention = output_attention
        self.use_attention = use_attention


        self.conv_skip = nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride)
        self.bn_skip = DomainSpecificBatchNorm1D(output_channel)

    def forward(self, x, domain_label='t'):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out,domain_label)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out,domain_label)

        if self.input_channel != self.output_channel:
            residual = self.conv_skip(x)
            residual = self.bn_skip(residual,domain_label)

        if self.use_attention:
            out_ca = self.ca(out)
            out = out_ca * out + residual
        else:
            out_ca = None
            out = out + residual
        out = self.activation(out)

        if self.output_attention:
            return out,out_ca
        else:
            return out

class ResNet1d(nn.Module):
    '''
    (batch_size, Channel, Seq_len) ——> (batch_size, embedding_length)
    (batch_size, C, 128) --> (batch_size, 128)
    '''
    def __init__(self, input_channel=3,embedding_length=256):    # (batch_size, C, 128)
        super(ResNet1d, self).__init__()
        self.embedding_length = embedding_length

        self.conv1 = nn.Conv1d(input_channel, 16, kernel_size=1, stride=1, padding=0)  # batch_size, 16, 128
        self.bn1 = DomainSpecificBatchNorm1D(16)
        self.activation = nn.ReLU()


        self.layer1 = ResNet1dBlock(16, 32, stride=1,use_attention=False)  # batch, 32, 128
        self.layer2 = ResNet1dBlock(32, 64, stride=2,use_attention=False)  # batch, 64, 64
        self.layer3 = ResNet1dBlock(64, 128, stride=2,use_attention=True)  # batch, 128, 32
        self.layer4 = ResNet1dBlock(128, 256, stride=2,use_attention=True)  # batch, 192, 16

        self.linear = nn.Linear(256 * 16, embedding_length)

        self.project_head = nn.ModuleList([
            nn.Linear(32 * 128, embedding_length),
            nn.Linear(64 * 64, embedding_length),
            nn.Linear(128 * 32, embedding_length),
            nn.Linear(256 * 16, embedding_length)
        ])


    def forward(self, x, domain_label='t'):
        out = self.conv1(x)
        out = self.bn1(out,domain_label)
        out = self.activation(out)

        out1, ca1 = self.layer1(out,domain_label)
        out2, ca2 = self.layer2(out1,domain_label)
        out3, ca3 = self.layer3(out2,domain_label)
        out4, ca4 = self.layer4(out3,domain_label)
        embedding = self.linear(out4.view(out4.size(0), -1))

        return embedding, [self.project_head[0](out1.view(out1.size(0), -1)),self.project_head[1](out2.view(out2.size(0), -1))], [ca1, ca2, ca3, ca4]

    def output_dim(self):
        return self.embedding_length

class Predictor(nn.Module):
    '''
    (batch_size, embedding_length) --> (batch_size,)
    (batch_size, 256) --> (batch_size, )
    '''
    def __init__(self,embedding_length=256):
        super(Predictor,self).__init__()
        self.predit = nn.Sequential(
            nn.Linear(embedding_length,128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self,x):
        out = self.predit(x)
        return out

if __name__ == '__main__':
    c_in = 32
    l_in = 128
    b_s = 32
    x = torch.rand((b_s,c_in,l_in))
    ca = ChannelAttention(input_channel=32)
    print(x.shape)
    atten = ca(x)
    print(atten.shape)
