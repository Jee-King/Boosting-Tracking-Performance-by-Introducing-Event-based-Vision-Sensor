import torch.nn as nn
import torch
import os
import torch.nn.functional as F



class ConvLSTM_qkv(nn.Module):



    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM_qkv, self).__init__()


        
        # for downsample

        self.conv3_cat = nn.Sequential(nn.Conv2d(in_channels= 3*3, out_channels=128, kernel_size=1, padding=0),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))

    def forward(self, e1, e2, e3):

        ecat = torch.cat([e1, e2, e3], dim=1)
        ecat = F.interpolate(ecat, size=[36,36])
        c_low = self.conv3_cat(ecat)
        c_high = self.conv4(c_low)
        
        return c_low, c_high


if __name__ == '__main__':
    channels = 3
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = ConvLSTM(input_dim=channels,
                 hidden_dim=[32],
                 kernel_size=(3, 3),
                 num_layers=1,
                 batch_first=True,
                 bias=True,
                 return_all_layers=True)
    model = model.cuda()
    var1 = torch.FloatTensor(60, 3, 288, 288).cuda()
    var2 = torch.FloatTensor(60, 3, 288, 288).cuda()
    var3 = torch.FloatTensor(60, 3, 288, 288).cuda()
    layer_output_list, last_state_list = model(var1, var2, var3)
    print(layer_output_list.shape)
    print(last_state_list.shape)