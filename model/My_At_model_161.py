"""
  Filename       [ My_At_model_161.py ]
  PackageName    [ AtJ_DH.model ]
  Synopsis       [ Self modified AtJ Model ] 
"""

from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models


class BottleneckDecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckDecoderBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(in_planes + 32)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(in_planes + 2 * 32)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(in_planes + 3 * 32)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn5 = nn.BatchNorm2d(in_planes + 4 * 32)
        self.relu5 = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm2d(in_planes + 5 * 32)
        self.relu6 = nn.ReLU(inplace=True)
        self.bn7 = nn.BatchNorm2d(inter_planes)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, 32, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            in_planes + 32, 32, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.conv3 = nn.Conv2d(
            in_planes + 2 * 32, 32, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.conv4 = nn.Conv2d(
            in_planes + 3 * 32, 32, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.conv5 = nn.Conv2d(
            in_planes + 4 * 32, 32, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.conv6 = nn.Conv2d(
            in_planes + 5 * 32, inter_planes, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.conv7 = nn.Conv2d(
            inter_planes, out_planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out3 = torch.cat([out2, out3], 1)
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out4 = torch.cat([out3, out4], 1)
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out5 = torch.cat([out4, out5], 1)
        out6 = self.conv6(self.relu6(self.bn6(out5)))
        out = self.conv7(self.relu7(self.bn7(out6)))

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        # out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        
        return torch.cat([x, out], 1)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=1, stride=1,
            padding=0, bias=False)

        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

        return F.upsample_nearest(out, scale_factor=2)

class Dense_encoder(nn.Module):
    def __init__(self, channel=3, ):
        super(Dense_encoder, self).__init__()

        dense=models.densenet161(pretrained=True)
        
        self.conv0 = dense.features.conv0
        self.norm0 = dense.features.norm0
        self.relu0 = dense.features.relu0
        self.pool0 = dense.features.pool0

        self.dense_block1 = dense.features.denseblock1
        self.trans_block1 = dense.features.transition1

        self.dense_block2 = dense.features.denseblock2
        self.trans_block2 = dense.features.transition2

        self.dense_block3 = dense.features.denseblock3
        self.trans_block3 = dense.features.transition3

        self.dense_block4 = BottleneckDecoderBlock(1056, 528)   # 512 -> 1056
        self.trans_block4 = TransitionBlock(1584, 256)          # 768 -> 1312
        # self.residual_block41 = ResidualBlock(128)              # Addition Blocks
        # self.residual_block42 = ResidualBlock(128)              # Addition Blocks
        # self.residual_block43 = ResidualBlock(128)              # Addition Blocks
        # self.residual_block44 = ResidualBlock(128)              # Addition Blocks
        # self.residual_block45 = ResidualBlock(128)              # Addition Blocks
        # self.residual_block46 = ResidualBlock(128)              # Addition Blocks

    def forward(self, x):
        # ---------------------------------------------------- #
        # Encoder: Input Shape is assumed as (3, 512, 512)     #
        # ---------------------------------------------------- #
        # 64 x 128 x 128
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))

        # 128 x 64 x 64
        x1 = self.dense_block1(x0)
        x1 = self.trans_block1(x1)

        # 256 x 32 x 32
        x2 = self.trans_block2(self.dense_block2(x1))

        # 896 x 16 x 16
        x3 = self.trans_block3(self.dense_block3(x2))

        # 256 x 32 x 32
        x4 = self.trans_block4(self.dense_block4(x3))

        # 256 x 32 x 32
        # x4 = self.residual_block41(x4)
        # x4 = self.residual_block42(x4)
        
        return x1, x2, x4

class Dense_decoder(nn.Module):
    def __init__(self, output_channel: int):
        super(Dense_decoder, self).__init__()
        ############# Block5-up 16-16 ##############
        self.dense_block5 = BottleneckDecoderBlock(640, 320)        # 384 -> 512
        self.trans_block5 = TransitionBlock(960, 160)               # 640 -> 768
        self.residual_block51 = ResidualBlock(160)                  # Addition Block
        self.residual_block52 = ResidualBlock(160)                  # Addition Block
        # self.residual_block53 = ResidualBlock(128)                  # Addition Block
        # self.residual_block54 = ResidualBlock(128)                  # Addition Block
        # self.residual_block55 = ResidualBlock(128)                  # Addition Block
        # self.residual_block56 = ResidualBlock(128)                  # Addition Block

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckDecoderBlock(352, 128)        # 256 -> 320
        self.trans_block6 = TransitionBlock(480, 64)                # 384 -> 448
        self.residual_block61 = ResidualBlock(64)                   # Addition Block
        self.residual_block62 = ResidualBlock(64)                   # Addition Block
        # self.residual_block63 = ResidualBlock(64)                   # Addition Block
        # self.residual_block64 = ResidualBlock(64)                   # Addition Block
        # self.residual_block65 = ResidualBlock(64)                   # Addition Block
        # self.residual_block66 = ResidualBlock(64)                   # Addition Block

        ############# Block7-up 64-64   ##############
        self.dense_block7 = BottleneckDecoderBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)
        self.residual_block71 = ResidualBlock(32)                   # Addition Block
        self.residual_block72 = ResidualBlock(32)                   # Addition Block
        self.residual_block73 = ResidualBlock(32)                   # Addition Block
        self.residual_block74 = ResidualBlock(32)                   # Addition Block
        self.residual_block75 = ResidualBlock(32)                   # Addition Block
        self.residual_block76 = ResidualBlock(32)                   # Addition Block

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8 = BottleneckDecoderBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)
        self.residual_block81 = ResidualBlock(16)                   # Addition Block
        self.residual_block82 = ResidualBlock(16)                   # Addition Block
        # self.residual_block83 = ResidualBlock(16)                   # Addition Block
        # self.residual_block84 = ResidualBlock(16)                   # Addition Block
        # self.residual_block85 = ResidualBlock(16)                   # Addition Block
        # self.residual_block86 = ResidualBlock(16)                   # Addition Block
        
        self.conv_refin = nn.Conv2d(19, 20, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm

        self.refine = nn.Sequential(
            nn.Conv2d(24, 20, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(20, 20, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(20, output_channel, kernel_size=7, stride=1, padding=3)
        )

        self.upsample = F.upsample_nearest
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1, x2, x4):
        # ---------------------------------------------------- #
        # Decoder:                                             # 
        #     Input Shape is assumed as                        #
        #       x1 : (3, 512, 512)                             #
        # ---------------------------------------------------- #
        x42 = torch.cat([x4, x2], 1)        # Concat

        # 512 x 32 x 32
        x5 = self.trans_block5(self.dense_block5(x42))
        x5 = self.residual_block51(x5)
        x5 = self.residual_block52(x5)
        x52 = torch.cat([x5, x1], 1)        # Concat

        # 320 x 64 x 64
        x6 = self.trans_block6(self.dense_block6(x52))
        x6 = self.residual_block61(x6)
        x6 = self.residual_block62(x6)

        # 64 x 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.residual_block71(x7)
        x7 = self.residual_block72(x7)

        # 128 x 128
        x8 = self.trans_block8(self.dense_block8(x7))
        x8 = self.residual_block81(x8)
        x8 = self.residual_block82(x8)
        x8 = torch.cat([x8, x], 1)          # Concat

        # 256 x 256
        x9 = self.relu(self.conv_refin(x8))
        
        # Multi-Size
        shape_out = x9.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)

        dehaze = self.refine(dehaze)

        return dehaze

class Dense_At(nn.Module):
    def __init__(self):
        super(Dense_At, self).__init__()

        self.encoder   = Dense_encoder()
        self.decoder_A = Dense_decoder(output_channel=3)
        self.decoder_T = Dense_decoder(output_channel=3)
        # self.decoder_J = Dense_decoder()

        self.convT1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.ResT   = ResidualBlock(32)
        self.convT  = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.sigT   = nn.Sigmoid()

    def forward(self, x):
        # ---------------------------------------------------- #
        # Encoder: Input Shape is assumed as (512, 512)        #
        # ---------------------------------------------------- #
        x1, x2, x4 = self.encoder(x)

        # ---------------------------------------------------- #
        # Decoder A : Input Shape is assumed as ...            #
        # ---------------------------------------------------- #
        A = self.decoder_A(x, x1, x2, x4)

        # ---------------------------------------------------- #
        # Decoder T : Input Shape is assumed as ...            #
        # ---------------------------------------------------- # 
        T = self.decoder_T(x, x1, x2, x4)
        T = self.sigT(self.convT(self.ResT(self.convT1(T))))
        T = torch.cat([T, T, T], 1)
        
        J = (x - A * (1 - T)) / T
        
        return J, A, T

class Dense_J(nn.Module):
    def __init__(self):
        super(Dense_J, self).__init__()

        self.encoder   = Dense_encoder()
        self.decoder_J = Dense_decoder()

    def forward(self, x):
        x1, x2, x4 = self.encoder(x)
        J_direct = self.decoder_J(x, x1, x2, x4)
   
        return J_direct

class Dense_AtJ(nn.Module):
    def __init__(self):
        super(Dense_AtJ, self).__init__()

        self.encoder   = Dense_encoder()
        self.decoder_A = Dense_decoder()
        self.decoder_T = Dense_decoder()
        self.decoder_J = Dense_decoder()

        self.convT1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.ResT   = ResidualBlock(32)
        self.convT  = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.sigT   = nn.Sigmoid()

    def forward(self, x):
        # ---------------------------------------------------- #
        # Encoder: Input Shape is assumed as (512, 512)        #
        # ---------------------------------------------------- #
        x1, x2, x4 = self.encoder(x)

        # ---------------------------------------------------- #
        # Decoder: Input Shape is assumed as (8, 8)            #
        # ---------------------------------------------------- #
        A = self.decoder_A(x, x1, x2, x4)
        T = self.decoder_T(x, x1, x2, x4)
        J_direct = self.decoder_J(x, x1, x2, x4)

        T = self.sigT(self.convT(self.ResT(self.convT1(T))))
        T = torch.cat([T, T, T], 1)
        
        J = (x - A * (1 - T)) / T
        
        return J, J_direct, A, T

def myModelParamsTransition(model_from, model_to):
    return

def main():
    x = torch.randn(1, 3, 512, 512)
    model = Dense_At()
    x, _, _ = model(x)

if __name__ == "__main__":
    main()
