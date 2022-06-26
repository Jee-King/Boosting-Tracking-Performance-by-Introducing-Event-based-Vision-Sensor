# -*- coding: utf-8 -*

from loguru import logger

import torch
from torchvision import models

from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)
import torch.nn as nn
import torch
import os

class Multi_Context(nn.Module):
    def __init__(self, inchannels):
        super(Multi_Context, self).__init__()
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels * 3, out_channels=inchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannels))

    def forward(self, x):
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x = torch.cat([x1,x2,x3], dim=1)
        x = self.conv2(x)
        return x

class Adaptive_Weight(nn.Module):
    def __init__(self, inchannels):
        super(Adaptive_Weight, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.inchannels = inchannels
        self.fc1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(inchannels//4, 1, kernel_size=1, bias=False)
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.avg(x)
        weight = self.relu1(self.fc1(x_avg))
        weight = self.relu2(self.fc2(weight))
        weight = self.sigmoid(weight)
        out = x * weight
        # with open('/home/iccd/1.txt', 'a') as f:
        #     f.write(str(weight.data)[11:17])
        #     f.write('\n')
        return out

class Counter_attention(nn.Module):
    def __init__(self, inchannels):
        super(Counter_attention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(inchannels))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(inchannels))
        # self.conv3 = nn.Sequential(nn.Conv2d(in_channels=inchannels*2, out_channels=inchannels, kernel_size=1),
        #                            nn.BatchNorm2d(inchannels))
        self.sig = nn.Sigmoid()
        self.mc1 = Multi_Context(inchannels)
        self.mc2 = Multi_Context(inchannels)
        self.ada_w1 = Adaptive_Weight(inchannels)
        self.ada_w2 = Adaptive_Weight(inchannels)
    def forward(self, assistant, present):

        mc1 = self.mc1(assistant)
        pr1 = present * self.sig(mc1)
        pr2 = self.conv1(present)
        pr2 = present * self.sig(pr2)
        out1 = pr1 + pr2 + present


        mc2 = self.mc2(present)
        as1 = assistant * self.sig(mc2)
        as2 = self.conv2(assistant)
        as2 = assistant * self.sig(as2)
        out2 = as1 + as2 + assistant


        out1 = self.ada_w1(out1)
        out2 = self.ada_w2(out2)
        out = out1 + out2

        # out = torch.cat([out1, out2], dim=1)
        # out = self.conv3(out)

        return out

class Counter_Guide(nn.Module):
    def __init__(self):
        super(Counter_Guide, self).__init__()
        self.counter_atten1 = Counter_attention(128)
        # self.counter_atten2 = Counter_attention(256)



    def forward(self, frame, event):
        out1 = self.counter_atten1(frame, event)

        return out1

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.softmax = nn.Softmax(dim=-1)
        self.ex = nn.Sequential(nn.Conv2d(self.input_dim, self.hidden_dim, 3, 1, 1),
                                     nn.BatchNorm2d(self.hidden_dim),
                                     nn.ReLU())

        self.sub = nn.Sequential(nn.BatchNorm2d(self.hidden_dim),
                                nn.ReLU())
        self.fusion = nn.Sequential(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1),
                                     nn.BatchNorm2d(self.hidden_dim),
                                     nn.ReLU())
        self.spatial_att = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1),
                                nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True))

        self.conv = nn.Conv2d(in_channels= 2 * self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)



    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        input_tensor = self.ex(input_tensor)
        m_batchsize, C, height, width = input_tensor.size()
        ### for h_t-i
        h_query = h_cur.view(m_batchsize, C, -1)
        h_key = h_cur.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(h_query, h_key)
        attention = self.softmax(energy)
        h_value = h_cur.view(m_batchsize, C, -1)
        h_out = torch.bmm(attention, h_value)
        h_out = h_out.view(m_batchsize, C, height, width)

        ### for X_t
        x_key = input_tensor.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(h_query, x_key)
        attention = self.softmax(energy)
        x_value = input_tensor.view(m_batchsize, C, -1)
        x_out = torch.bmm(attention, x_value)
        x_out = x_out.view(m_batchsize, C, height, width)

        # combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        # print(x_out.size(), h_out.size())
        sub = self.sub(x_out - h_out)
        sub = self.fusion(sub)
        sub = torch.mean(sub,1).unsqueeze(1)
        sub_att = torch.sigmoid(self.spatial_att(sub))
        x_out = sub_att * x_out
        h_out = sub_att * h_out

        combined = torch.cat([x_out, h_out], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM_qkv(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM_qkv, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        
        # for downsample
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(3),
                                   nn.ReLU(inplace=True))
        # self.conv3_cat = nn.Sequential(nn.Conv2d(in_channels=sum(self.hidden_dim), out_channels=128, kernel_size=1, padding=0),
        #                            nn.BatchNorm2d(128),
        #                            nn.ReLU(inplace=True))
        self.conv3_cat = nn.Sequential(nn.Conv2d(in_channels=self.hidden_dim[-1]*3, out_channels=128, kernel_size=1, padding=0),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))
        # self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(inplace=True))

    def forward(self, e1, e2, e3, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        e1 = self.conv3(self.conv2(self.conv1(e1)))
        e2 = self.conv3(self.conv2(self.conv1(e2)))
        e3 = self.conv3(self.conv2(self.conv1(e3)))

        e1 = torch.unsqueeze(e1, 1)
        e2 = torch.unsqueeze(e2, 1)
        e3 = torch.unsqueeze(e3, 1)
        
        input_tensor = torch.cat([e1,e2,e3],dim=1)
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        c_out = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            c_output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
                c_output_inner.append(c)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
            c_out.append(c)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        # downsample and output c_out
        # c_cat = torch.cat([c_out[0], c_out[1]], dim=1)
        c_cat = torch.cat([c_output_inner[0], c_output_inner[1], c_output_inner[2]], dim=1)

        # c_cat = self.conv2(self.conv1(c_cat))
        # c_low = self.conv3(c_cat)
        c_low = self.conv3_cat(c_cat)
        # c_high = self.conv4(c_low)
        
        # return c_low, c_high
        return c_low
        #return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

torch.set_printoptions(precision=8)

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的后1层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-4])
        # self.Linear_layer = nn.Linear(512, 2)  # 加上一层参数修改好的全连接层

    def forward(self, x):
        x = self.resnet_layer(x)
        # x = x.view(x.size(0), -1)
        # x = self.Linear_layer(x)
        return x

@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class SiamTrack(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(pretrain_model_path="",
                                head_width=128,
                                conv_weight_std=0.01,
                                neck_conv_bias=[True, True, True, True],
                                corr_fea_output=False,
                                trt_mode=False,
                                trt_fea_model_path="",
                                trt_track_model_path="",
                                amp=False)

    support_phases = ["train", "feature", "track", "freeze_track_fea"]

    def __init__(self, backbone, head, loss=None):
        super(SiamTrack, self).__init__()
        self.basemodel = Net(models.resnet18(pretrained=True))
        self.head = head
        self.loss = loss
        channels = 3
        self.model = ConvLSTM_qkv(input_dim=channels,
                 hidden_dim=[32,32],
                 kernel_size=(3, 3),
                 num_layers=2,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False).cuda()
        self.fusion =  Counter_Guide().cuda()
        self.trt_fea_model = None
        self.trt_track_model = None
        self._phase = "train"

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def train_forward(self, training_data):
        target_img = training_data["im_z"]
        search_img = training_data["im_x"]
        ex1,ex2,ex3 = training_data["em_x1"],training_data["em_x2"], training_data["em_x3"]
        ez1,ez2,ez3 = training_data["em_z1"],training_data["em_z2"], training_data["em_z3"]

        # backbone feature
        ff_z = self.basemodel(target_img)
        ff_x = self.basemodel(search_img)
        e_z = self.model(ez1,ez2,ez3)

        e_x = self.model(ex1,ex2,ex3)
        f_z = self.fusion(ff_z,e_z)
        f_x = self.fusion(ff_x,e_x)

        # feature adjustment
        c_z_k = self.c_z_k(f_z)
        r_z_k = self.r_z_k(f_z)
        c_x = self.c_x(f_x)
        r_x = self.r_x(f_x)
        # feature matching
        r_out = xcorr_depthwise(r_x, r_z_k)
        c_out = xcorr_depthwise(c_x, c_z_k)
        # head
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
            c_out, r_out)
        predict_data = dict(
            cls_pred=fcos_cls_score_final,
            ctr_pred=fcos_ctr_score_final,
            box_pred=fcos_bbox_final,
        )
        if self._hyper_params["corr_fea_output"]:
            predict_data["corr_fea"] = corr_fea
        return predict_data

    def instance(self, img):
        f_z = self.basemodel(img)
        # template as kernel
        c_x = self.c_x(f_z)
        self.cf = c_x

    def forward(self, *args, phase=None):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        if phase is None:
            phase = self._phase
        # used during training
        if phase == 'train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])

        # used for template feature extraction (normal mode)
        elif phase == 'feature':
            target_img, ez1,ez2,ez3 = args
            if self._hyper_params["trt_mode"]:
                # extract feature with trt model
                out_list = self.trt_fea_model(target_img)
            else:
                # backbone feature
                e_z = self.model(ez1,ez2,ez3)


                ff_z = self.basemodel(target_img)
                f_z = self.fusion(ff_z,e_z)                # template as kernel
                c_z_k = self.c_z_k(f_z)
                r_z_k = self.r_z_k(f_z)
                # output
                out_list = [c_z_k, r_z_k]
        # used for template feature extraction (trt mode)
        elif phase == "freeze_track_fea":
            search_img, = args
            # backbone feature
            f_x = self.basemodel(search_img)
            # feature adjustment
            c_x = self.c_x(f_x)
            r_x = self.r_x(f_x)
            # head
            return [c_x, r_x]
        # [Broken] used for template feature extraction (trt mode)
        #   currently broken due to following issue of "torch2trt" package
        #   c.f. https://github.com/NVIDIA-AI-IOT/torch2trt/issues/251
        elif phase == "freeze_track_head":
            c_out, r_out = args
            # head
            outputs = self.head(c_out, r_out, 0, True)
            return outputs
        # used for tracking one frame during test
        elif phase == 'track':
            if len(args) == 6:
                search_img,ex1,ex2,ex3, c_z_k, r_z_k = args
                if self._hyper_params["trt_mode"]:
                    c_x, r_x = self.trt_track_model(search_img)
                else:
                    # backbone feature
                    ff_x = self.basemodel(search_img)
                    e_x = self.model(ex1,ex2,ex3)
                    f_x = self.fusion(ff_x,e_x)
                    # feature adjustment
                    c_x = self.c_x(f_x)
                    r_x = self.r_x(f_x)
            # elif len(args) == 4:
            #     # c_x, r_x already computed
            #     c_z_k, r_z_k, c_x, r_x = args
            else:
                raise ValueError("Illegal args length: %d" % len(args))

            # feature matching
            r_out = xcorr_depthwise(r_x, r_z_k)
            c_out = xcorr_depthwise(c_x, c_z_k)
            # head
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
                c_out, r_out, search_img.size(-1))
            # apply sigmoid
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
            # register extra output
            extra = dict(c_x=c_x, r_x=r_x, corr_fea=corr_fea)
            self.cf = c_x
            # output
            out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    def update_params(self):
        r"""
        Load model parameters
        """
        self._make_convs()
        self._initialize_conv()
        super().update_params()
        if self._hyper_params["trt_mode"]:
            logger.info("trt mode enable")
            from torch2trt import TRTModule
            self.trt_fea_model = TRTModule()
            self.trt_fea_model.load_state_dict(
                torch.load(self._hyper_params["trt_fea_model_path"]))
            self.trt_track_model = TRTModule()
            self.trt_track_model.load_state_dict(
                torch.load(self._hyper_params["trt_track_model_path"]))
            logger.info("loading trt model succefully")

    def _make_convs(self):
        head_width = self._hyper_params['head_width']

        # feature adjustment
        self.r_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.c_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
