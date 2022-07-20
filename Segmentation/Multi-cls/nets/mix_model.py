import torch
import torch.nn as nn
from nets.UNetpp import NestedUNet as Upp
from nets.unet___resnetenco_deco import Resnet34 as Res34Upp
from nets.TransUNetzip import TransUnet


class MixUpp(nn.Module):
    def __init__(self, img_size=256, in_channel=1, out_channel=3, pretrained=True):
        super(MixUpp, self).__init__()
        # Upp to predict 170 pixel
        self.model1 = Upp(num_classes=1, input_channels=in_channel)
        if pretrained:
            self.model1.load_state_dict(
                torch.load('model/Uppto170+TransORUpp_85_255/Upp256_to170_Dice__model_Adam_cos__epoch15.pth'))

        # Upp to predict 85, 255 pixel
        self.model2 = Upp(num_classes=2, input_channels=in_channel)
        if pretrained:
            self.model2.load_state_dict(
                torch.load('model/Uppto170+TransORUpp_85_255/Upp256_to85+255_Dice__model_Adam_cos__epoch15.pth'))

        # TransU to predict 85, 255 pixel
        # self.model2 = TransUnet(img_dim=img_size, in_channels=in_channel,
        #                         vit_blocks=6, vit_dim_linear_mhsa_block=img_size, classes=out_channel-1)
        # if pretrained:
        #     self.model2.load_state_dict(
        #         torch.load('model/Uppto170+TransORUpp_85_255/TransU_to85+255_Dice__model_Adam_lam__epoch20.pth'))

    def forward(self, x):
        out_170 = self.model1(x)
        out_85_255 = self.model2(x)
        # print(out_170.shape)
        # print(out_85_255.shape)
        out_85_255 = torch.split(out_85_255, 1, dim=1)
        out_85 = out_85_255[0]
        out_255 = out_85_255[1]
        # print(out_85.shape)
        out = torch.cat((out_85, out_170, out_255), dim=1)
        # print(out.shape)
        return out


class TripalsUpp(nn.Module):
    def __init__(self, img_size=256, in_channel=1, out_channel=3, pretrained=True):
        super(TripalsUpp, self).__init__()
        # Upp to predict 85 pixel
        self.model1 = Upp(num_classes=1, input_channels=in_channel)
        if pretrained:
            self.model1.load_state_dict(
                torch.load('model/UppUpp85_170_255/Upp256to85_Dice__model_Adam_cos__epoch15.pth'))

        # Upp to predict 170 pixel
        self.model2 = Upp(num_classes=1, input_channels=in_channel)
        if pretrained:
            self.model2.load_state_dict(
                torch.load('model/UppUpp85_170_255/Upp256_to170_Dice__model_Adam_cos__epoch15.pth'))

        # Upp to predict 255 pixel
        self.model3 = Upp(num_classes=1, input_channels=in_channel)
        if pretrained:
            self.model3.load_state_dict(
                torch.load('model/UppUpp85_170_255/Upp256to255_Dice__model_Adam_cos__epoch15.pth'))

    def forward(self, x):
        out_85 = self.model1(x)
        out_170 = self.model2(x)
        out_255 = self.model3(x)
        out = torch.cat((out_85, out_170, out_255), dim=1)
        return out


class MixRes34Upp(nn.Module):
    def __init__(self, img_size=256, in_channel=1, out_channel=3, pretrained=True):
        super(MixRes34Upp, self).__init__()
        # Upp to predict 170 pixel
        self.model1 = Res34Upp(inchannels=in_channel, outcls=1)
        if pretrained:
            self.model1.load_state_dict(
                torch.load('model/Res34Uppmore/Res34Upp256_to170_Dice__model_Adam_cos__epoch15.pth'))

        # Upp to predict 85, 255 pixel
        self.model2 = Res34Upp(inchannels=in_channel, outcls=2)
        if pretrained:
            self.model2.load_state_dict(
                torch.load('model/Res34Uppmore/Res34Upp256_to85_255_Dice__model_Adam_cos__epoch15.pth'))

    def forward(self, x):
        out_170 = self.model1(x)
        out_85_255 = self.model2(x)
        out_85_255 = torch.split(out_85_255, 1, dim=1)
        out_85 = out_85_255[0]
        out_255 = out_85_255[1]
        out = torch.cat((out_85, out_170, out_255), dim=1)
        return out


class AddSmallUpp(nn.Module):
    def __init__(self, img_size=256, in_channel=1, out_channel=3, pretrained=True):
        super(AddSmallUpp, self).__init__()
        # Upp to normal
        self.model1 = MixUpp(img_size=img_size, in_channel=in_channel, out_channel=out_channel, pretrained=False)
        if pretrained:
            self.model1.load_state_dict(
                torch.load('model/Uppto170+TransORUpp_85_255/UppUpp_Dice__model_Adam_cos__epoch15.pth'))

        # Upp to small
        self.model2 = Upp(num_classes=out_channel, input_channels=in_channel)
        if pretrained:
            self.model2.load_state_dict(
                torch.load('model/Uppsmall_Dice__model_Adam_cos__epoch15.pth'))

        # self.fc = nn.Linear(in_features=img_size*img_size*out_channel*2, out_features=img_size*img_size*out_channel)

    def forward(self, x):
        out_normal = self.model1(x)
        out_small = self.model2(x)
        out_mix = out_normal + out_small
        # print(out_mix.shape)
        # out_squeeze = self.fc(out_mix)
        # out = out_squeeze.reshape(8, 3, 256, 256)
        return out_mix
