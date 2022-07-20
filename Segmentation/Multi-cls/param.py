import torch
from nets import Upp, TransUnet, MixUpp, AddSmallUpp, TripalsUpp
from nets import Deeplabv3plus_res101, Deeplabv3plus_res50, DDRNet, BiSeNetV2, HRNet, FCN8, SegNet
from nets import PSPNet
from nets import Res34Upp, MixRes34Upp
from nets import UNet_3Plus_DeepSup, UNet_3Plus
from multi_score import DiceLoss, MultipleOutputLoss2
from losses_pytorch.dice_loss import DC_and_topk_loss

img_cha = 1
crop_size = 128
img_size = 256
img_class = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Dice Loss trainable
# Run uni_train.py

# model = 'Upp256_to85_255_'
# create_model = Upp(n_channels=img_cha, num_classes=img_class)
# model = 'Upp256to85_'
# model = 'Upp256to170_'
# model = 'Upp256to255_'
# create_model = Upp(n_channels=img_cha, num_classes=img_class)

# Run kfold_train.py
model = 'UppUpp256_'
# create_model = MixUpp(img_size=img_size, in_channel=img_cha, out_channel=img_class, pretrained=True)
create_model = MixUpp(img_size=img_size, in_channel=img_cha, out_channel=img_class, pretrained=False)
create_model.load_state_dict(torch.load(
    'model/Uppto170+TransORUpp_85_255/UppUpp_Dice__model_Adam_cos__epoch15.pth', map_location=device))
# model = 'UppUppTripal256_'
# create_model = TripalsUpp(img_size=img_size, in_channel=img_cha, out_channel=img_class, pretrained=True)
# create_model = MixUpp(img_size=img_size, in_channel=img_cha, out_channel=img_class, pretrained=False)


# model = 'Res34Upp256_to85_255_'
# create_model = Res34Upp(inchannels=img_cha, outcls=img_class)
# model = 'Res34Upp256_to170_'
# create_model = Res34Upp(inchannels=img_cha, outcls=img_class)

# Run kfold_train.py
# model = 'Res34UppUpp256more_'
# create_model = MixRes34Upp(img_size=img_size, in_channel=img_cha, out_channel=img_class, pretrained=True)

# Run small_train.py
# model = 'Uppsmall_'
# create_model = Upp(n_channels=img_cha, num_classes=img_class)
# Run kfold_train.py
# model = 'UppUpp+small_'
# create_model = AddSmallUpp(pretrained=False)

# model = 'TransU_to85+255_'
# create_model = TransUnet(in_channels=img_cha, img_dim=img_size, vit_blocks=6, vit_dim_linear_mhsa_block=img_size,
#                          classes=img_class)
# model = 'UppUpp224crop112_'
# # create_model = MixUpp(img_size=img_size, in_channel=img_cha, out_channel=img_class, pretrained=True)
# create_model = MixUpp(img_size=img_size, in_channel=img_cha, out_channel=img_class, pretrained=False)

# model = 'Deeplabv3+res101_'
# create_model = Deeplabv3plus_res101(num_classes=img_class)

# model = 'DDRNet_'
# create_model = DDRNet(num_classes=img_class)

# model = 'BiSeNetV2_'
# create_model = BiSeNetV2(num_classes=img_class)

# model = 'HRNet_'
# create_model = HRNet(num_classes=img_class)

# model = 'FCN8_'
# create_model = FCN8(num_classes=3)

# model = 'FCNRes101_'
# create_model = FCN_ResNet(num_classes=3, backbone='resnet101')

# model = 'Uppp_'
# create_model = UNet_3Plus(in_channels=img_cha, n_classes=img_class)

# criterion = DiceLoss()
# loss = 'Dice_'

criterion = DC_and_topk_loss(soft_dice_kwargs={}, ce_kwargs={})
loss = 'Dice_Topk_'

# criterion = MultipleOutputLoss2(loss=DiceLoss())
# loss = 'MultiDice_'

opt_type = 'Adam_cos_'
epochs = 15

model_pth = './model/' + model + loss + '_model_' + opt_type + '_epoch' + str(epochs) + '.pth'
