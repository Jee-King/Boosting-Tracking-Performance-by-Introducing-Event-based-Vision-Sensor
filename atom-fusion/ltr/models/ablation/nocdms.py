import torch,os
import torch.nn as nn
from torch.nn.parameter import Parameter


class Counter_Guide(nn.Module):
    def __init__(self):
        super(Counter_Guide, self).__init__()

    def forward(self, frame1, frame2, event1, event2):
        out1 = frame1 + event1
        out2 = frame2 + event2

        return out1, out2


if __name__ == '__main__':
    net = Counter_Guide()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = net.cuda()

    var1 = torch.FloatTensor(10, 128, 36, 36).cuda()
    var2 = torch.FloatTensor(10, 256, 18, 18).cuda()
    var3 = torch.FloatTensor(10, 128, 36, 36).cuda()
    var4 = torch.FloatTensor(10, 256, 18, 18).cuda()
    # var = Variable(var)

    out1, out2 = net(var1, var2, var3, var4)

    print('*************')
    print(out1.shape, out2.shape)
