from nets.UNetfamily import U_Net, R2U_Net, AttU_Net, R2AttU_Net, Dense_Unet, LadderNet, init_weights
from nets.UNetpp import NestedUNet as Upp
from .TransUNetzip import TransUnet
from nets.mix_model import MixUpp, TripalsUpp
from nets.mix_model import AddSmallUpp
from nets.SegNets.DeeplabV3Plus import Deeplabv3plus_res101, Deeplabv3plus_res50
from nets.SegNets.DDRNet import DDRNet
from nets.SegNets.BiSeNetV2 import BiSeNetV2
from nets.SegNets.HRNet import HighResolutionNet as HRNet
from nets.SegNets.FCN8s import FCN as FCN8
# from nets.SegNets.FCN_ResNet import FCN_ResNet
from nets.SegNets.SegNet import SegNet
from nets.SegNets.PSPNet.pspnet import PSPNet
from nets.unet___resnetenco_deco import Resnet34 as Res34Upp
from nets.mix_model import MixRes34Upp
from nets.UNetppp.UNet_3Plus import UNet_3Plus_DeepSup, UNet_3Plus
