import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from . import block as B
from . import spectral_norm as SN
from . import adaptive_norm as AN

####################
# Generator
####################


class SRCNN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, norm_type='batch', act_type='relu', mode='CNA', ada_ksize=None):
        super(SRCNN, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=9, norm_type=norm_type, act_type=act_type, mode=mode
                                , ada_ksize=ada_ksize)
        mapping_conv = B.conv_block(nf, nf // 2, kernel_size=1, norm_type=norm_type, act_type=act_type,
                                    mode=mode, ada_ksize=ada_ksize)
        HR_conv = B.conv_block(nf // 2, out_nc, kernel_size=5, norm_type=norm_type, act_type=None,
                               mode=mode, ada_ksize=ada_ksize)

        self.model = B.sequential(fea_conv, mapping_conv, HR_conv)

    def forward(self, x):
        x = self.model(x)
        return x


class ARCNN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, norm_type='batch', act_type='relu', mode='CNA', ada_ksize=None):
        super(ARCNN, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=9, norm_type=norm_type, act_type=act_type, mode=mode
                                , ada_ksize=ada_ksize)
        conv1 = B.conv_block(nf, nf // 2, kernel_size=7, norm_type=norm_type, act_type=act_type,
                             mode=mode, ada_ksize=ada_ksize)
        conv2 = B.conv_block(nf // 2, nf // 4, kernel_size=1, norm_type=norm_type, act_type=act_type,
                             mode=mode, ada_ksize=ada_ksize)
        HR_conv = B.conv_block(nf // 4, out_nc, kernel_size=5, norm_type=norm_type, act_type=None,
                               mode=mode, ada_ksize=ada_ksize)

        self.model = B.sequential(fea_conv, conv1, conv2, HR_conv)

    def forward(self, x):
        x = self.model(x)
        return x


class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
             *upsampler,HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


class ModulateSRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='sft', act_type='relu',
                 mode='CNA', res_scale=1, upsample_mode='upconv', gate_conv_bias=True, ada_ksize=None):
        super(ModulateSRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, stride=1)
        resnet_blocks = [B.TwoStreamSRResNet(nf, nf, nf, norm_type=norm_type, act_type=act_type,
                         mode=mode, res_scale=res_scale, gate_conv_bias=gate_conv_bias,
                                             ada_ksize=ada_ksize, input_dim=in_nc) for _ in range(nb)]

        self.LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode=mode)
        if norm_type == 'sft':
            self.LR_norm = AN.GateNonLinearLayer(in_nc, conv_bias=gate_conv_bias)
        elif norm_type == 'sft_conv':
            self.LR_norm = AN.MetaLayer(in_nc, conv_bias=gate_conv_bias, kernel_size=ada_ksize)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]


        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.norm_branch = B.sequential(*resnet_blocks)
        self.HR_branch = B.sequential(*upsampler,HR_conv0, HR_conv1)

        self.CondNet = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),   #out_cnn_e=3, first number
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 3, 1, 1),
        )

    def forward(self, x):
        #cond = self.CondNet(x[1])
        fea = self.fea_conv(x[0])
        fea_res_block, _ = self.norm_branch((fea, x[1]))
        fea_LR = self.LR_conv(fea_res_block)
        res = self.LR_norm((fea_LR, x[1]))
        out = self.HR_branch(fea+res)
        return out

#noise_blur concat
class ModulateSRResNet_one(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='sft', act_type='relu',
                 mode='CNA', res_scale=1, upsample_mode='upconv', gate_conv_bias=True, ada_ksize=None):
        super(ModulateSRResNet_one, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, stride=1)
        resnet_blocks = [B.TwoStreamSRResNet(nf, nf, nf, norm_type=norm_type, act_type=act_type,
                         mode=mode, res_scale=res_scale, gate_conv_bias=gate_conv_bias,
                                             ada_ksize=ada_ksize, input_dim=6) for _ in range(nb)]

        self.LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode=mode)
        if norm_type == 'sft':
            self.LR_norm = AN.GateNonLinearLayer(in_nc=6, conv_bias=gate_conv_bias)
        elif norm_type == 'sft_conv':
            self.LR_norm = AN.MetaLayer(in_nc=6, conv_bias=gate_conv_bias, kernel_size=ada_ksize)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]


        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.norm_branch = B.sequential(*resnet_blocks)
        self.HR_branch = B.sequential(*upsampler,HR_conv0, HR_conv1)


        # x[0]:img  x[1]:noise_blur_est
    def forward(self, x):
        #cond = self.CondNet(x[1])
        fea = self.fea_conv(x[0])
        fea_res_block, _ = self.norm_branch((fea, x[1]))
        fea_LR = self.LR_conv(fea_res_block)
        res = self.LR_norm((fea_LR, x[1]))
        out = self.HR_branch(fea+res)
        return out

#noise_blur add  it's need to change
class ModulateSRResNet_two(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='sft', act_type='relu',
                 mode='CNA', res_scale=1, upsample_mode='upconv', gate_conv_bias=True, ada_ksize=None):
        super(ModulateSRResNet_two, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, stride=1)
        resnet_blocks = [B.TwoStreamSRResNet(nf, nf, nf, norm_type=norm_type, act_type=act_type,
                         mode=mode, res_scale=res_scale, gate_conv_bias=gate_conv_bias,
                                             ada_ksize=ada_ksize, input_dim=in_nc) for _ in range(nb)]

        self.LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode=mode)
        if norm_type == 'sft':
            self.LR_norm0 = AN.GateNonLinearLayer(in_nc, conv_bias=gate_conv_bias)
        elif norm_type == 'sft_conv':
            self.LR_norm0 = AN.MetaLayer(in_nc, conv_bias=gate_conv_bias, kernel_size=ada_ksize)
        if norm_type == 'sft':
            self.LR_norm1 = AN.GateNonLinearLayer(in_nc, conv_bias=gate_conv_bias)
        elif norm_type == 'sft_conv':
            self.LR_norm1 = AN.MetaLayer(in_nc, conv_bias=gate_conv_bias, kernel_size=ada_ksize)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]


        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.norm_branch = B.sequential(*resnet_blocks)
        self.HR_branch = B.sequential(*upsampler,HR_conv0, HR_conv1)

        # x[0]:img  x[1]:noise_est  x[2]:blur_est
    def forward(self, x):
        fea = self.fea_conv(x[0])
        fea_res_block, _ = self.norm_branch((fea,(x[1][0],x[1][1])))
        fea_LR = self.LR_conv(fea_res_block)
        res = self.LR_norm0((fea_LR, x[1][0]))
        res=self.LR_norm1((res,x[1][1]))
        out = self.HR_branch(fea+res)
        return out


class noise_subnet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(noise_subnet, self).__init__()
        self.CNN_n = nn.Sequential(
            nn.Conv2d(in_nc, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, out_nc, 3, 1, 1)
        )
    def forward(self, x):
        noise_est = self.CNN_n(x)
        return noise_est

class blur_subnet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(blur_subnet, self).__init__()
        self.CNN_b = nn.Sequential(
            nn.Conv2d(in_nc, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, out_nc, 3, 1, 1)
        )

    def forward(self, x):
        blur_est = self.CNN_b(x)
        return blur_est




class DenoiseResNet(nn.Module):
    """
    jingwen's addition
    denoise Resnet
    """
    def __init__(self, in_nc, out_nc, nf, nb, upscale=1, norm_type='batch', act_type='relu',
                 mode='CNA', res_scale=1, upsample_mode='upconv', ada_ksize=None, down_scale=2,
                 fea_norm=None, upsample_norm=None):
        super(DenoiseResNet, self).__init__()
        n_upscale = int(math.log(down_scale, 2))
        if down_scale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=fea_norm, act_type=None, stride=down_scale,
                                ada_ksize=ada_ksize)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,
                         mode=mode, res_scale=res_scale, ada_ksize=ada_ksize) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode
                               , ada_ksize=ada_ksize)
        # LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode=mode
        #                        , ada_ksize=ada_ksize)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)

        if down_scale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type, norm_type=upsample_norm, ada_ksize=ada_ksize)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type, norm_type=upsample_norm, ada_ksize=ada_ksize) for _ in range(n_upscale)]

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=upsample_norm, act_type=act_type, ada_ksize=ada_ksize)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=upsample_norm, act_type=None, ada_ksize=ada_ksize)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),
                                  *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


class ModulateDenoiseResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=1, norm_type='sft', act_type='relu',
                 mode='CNA', res_scale=1, upsample_mode='upconv', gate_conv_bias=True, ada_ksize=None):
        super(ModulateDenoiseResNet, self).__init__()

        self.fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, stride=2)
        resnet_blocks = [B.TwoStreamSRResNet(nf, nf, nf, norm_type=norm_type, act_type=act_type,
                         mode=mode, res_scale=res_scale, gate_conv_bias=gate_conv_bias,
                                             ada_ksize=ada_ksize, input_dim=in_nc) for _ in range(nb)]

        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode=mode)
        if norm_type == 'sft':
            LR_norm = AN.GateNonLinearLayer(in_nc, conv_bias=gate_conv_bias)
        elif norm_type == 'sft_conv':
            LR_norm = AN.MetaLayer(in_nc, conv_bias=gate_conv_bias, kernel_size=ada_ksize)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        upsampler = upsample_block(nf, nf, act_type=act_type)
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.norm_branch = B.sequential(*resnet_blocks)
        self.LR_conv = LR_conv
        self.LR_norm = LR_norm
        self.HR_branch = B.sequential(upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        fea = self.fea_conv(x[0])
        fea_res_block, _ = self.norm_branch((fea, x[1]))
        fea_LR = self.LR_conv(fea_res_block)
        res = self.LR_norm((fea_LR, x[1]))
        out = self.HR_branch(fea+res)
        return out


class NoiseSubNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, norm_type='batch', act_type='relu', mode='CNA'):
        super(NoiseSubNet, self).__init__()
        degration_block = [B.conv_block(in_nc, nf, kernel_size=3, norm_type=norm_type, act_type=act_type, mode=mode)]
        degration_block.extend([B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type, mode=mode)
                                for _ in range(15)])
        degration_block.append(B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None, mode=mode))
        self.degration_block = B.sequential(*degration_block)

    def forward(self, x):
        deg_estimate = self.degration_block(x)
        return deg_estimate


class CondDenoiseResNet(nn.Module):
    """
    jingwen's addition
    denoise Resnet
    """

    def __init__(self, in_nc, out_nc, nf, nb, upscale=1, res_scale=1, down_scale=2, num_classes=1, ada_ksize=None
                 ,upsample_mode='upconv', act_type='relu', norm_type='cond_adaptive_conv_res'):
        super(CondDenoiseResNet, self).__init__()
        n_upscale = int(math.log(down_scale, 2))
        if down_scale == 3:
            n_upscale = 1

        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=down_scale, padding=1)
        resnet_blocks = [B.CondResNetBlock(nf, nf, nf, num_classes=num_classes, ada_ksize=ada_ksize,
                                           norm_type=norm_type, act_type=act_type) for _ in range(nb)]
        self.resnet_blocks = B.sequential(*resnet_blocks)
        self.LR_conv = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        if norm_type == 'cond_adaptive_conv_res':
            self.cond_adaptive = AN.CondAdaptiveConvResNorm(nf, num_classes=num_classes)
        elif norm_type == "interp_adaptive_conv_res":
            self.cond_adaptive = AN.InterpAdaptiveResNorm(nf, ada_ksize)
        elif norm_type == "cond_instance":
            self.cond_adaptive = AN.CondInstanceNorm2d(nf, num_classes=num_classes)
        elif norm_type == "cond_transform_res":
            self.cond_adaptive = AN.CondResTransformer(nf, ada_ksize, num_classes=num_classes)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)

        if down_scale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.upsample = B.sequential(*upsampler, HR_conv0, HR_conv1)

    def forward(self, x, y):
        # the first feature extraction
        fea = self.fea_conv(x)
        fea1, _ = self.resnet_blocks((fea, y))
        fea2 = self.LR_conv(fea1)
        fea3 = self.cond_adaptive(fea2, y)
        # res
        out = self.upsample(fea3 + fea)
        return out


class AdaptiveDenoiseResNet(nn.Module):
    """
    jingwen's addition
    adabn
    """
    def __init__(self, in_nc, nf, nb, upscale=1, res_scale=1, down_scale=2):
        super(AdaptiveDenoiseResNet, self).__init__()

        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=down_scale, padding=1)
        resnet_blocks = [B.AdaptiveResNetBlock(nf, nf, nf, res_scale=res_scale) for _ in range(nb)]
        self.resnet_blocks = B.sequential(*resnet_blocks)
        self.LR_conv = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(nf, affine=True, track_running_stats=True, momentum=0)

    def forward(self, x):
        fea_list = [self.fea_conv(data.unsqueeze_(0)) for data in x]
        fea_resblock_list = self.resnet_blocks(fea_list)
        fea_LR_list = [self.LR_conv(fea) for fea in fea_resblock_list]
        fea_mean, fea_var = B.computing_mean_variance(fea_LR_list)

        batch_norm_dict = self.batch_norm.state_dict()
        batch_norm_dict['running_mean'] = fea_mean
        batch_norm_dict['running_var'] = fea_var
        self.batch_norm.load_state_dict(batch_norm_dict)
        return None


