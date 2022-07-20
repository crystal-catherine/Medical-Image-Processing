import os

import PIL.Image
import cv2
import glob
import numpy as np
import param
from numpy import uint8
from torchvision import transforms
from PIL import Image
from multi_score import *
from nets import Upp, TransUnet

if __name__ == "__main__":
    # net = TransUnet(in_channels=1, img_dim=256, vit_blocks=6, vit_dim_linear_mhsa_block=256,
    #                 classes=2)
    net = Upp(n_channels=1, num_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=param.device)
    # 加载模型参数
    net.load_state_dict(
        torch.load('model/Upp256to255_Dice__model_Adam_cos__epoch15.pth',
                   map_location=param.device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = './Heart Data/training'

    # 遍历素有图片
    for cls in os.listdir(tests_path):
        # DCM/HCM/NOR
        patients_img = os.path.join(tests_path, cls, "png", "Image")
        patients_lab = os.path.join(tests_path, cls, "png", "Label")
        patients_sav = os.path.join(tests_path, cls, "png", "Result_uni")
        for patient in os.listdir(patients_img):
            # 01..15, take 15 for test
            if patient == '15':
                img_folder = os.path.join(patients_img, patient)
                lab_folder = os.path.join(patients_lab, patient)
                save_folder = os.path.join(patients_sav, patient)
            else:
                continue

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for num in os.listdir(img_folder):
                img_path = os.path.join(img_folder, num)
                lab_path = os.path.join(lab_folder, num.replace("image", "label"))
                save_name = os.path.join(save_folder, num)
                print(save_name)

                if param.img_cha == 1:
                    img = Image.open(img_path).convert('L')
                else:
                    img = Image.open(img_path).convert('RGB')

                w, h = img.size
                data_transform = transforms.Compose([transforms.Resize((param.img_size, param.img_size)),
                                                     transforms.ToTensor()])
                data_transform2 = transforms.Compose([transforms.CenterCrop((param.crop_size, param.crop_size)),
                                                      transforms.Resize((param.img_size, param.img_size)),
                                                      transforms.ToTensor()])
                img = data_transform2(img)
                img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
                img = img.to(device=param.device)

                # 预测
                pre = net(img)
                # 提取结果
                pre = np.array(pre.data.cpu())[0]
                # 处理结果
                pre_pro = np.zeros((h, w), dtype=uint8)

                pre_pro1 = cv2.resize(pre[0], (w, h))
                pre_pro[pre_pro1 >= 0.5] = 255
                pre_pro[pre_pro1 < 0.5] = 0

                # pre_pro3 = cv2.resize(pre[1], (w, h))
                # pre_pro[pre_pro3 >= 0.5] = 255

                # pre_pro2 = cv2.resize(pre[0], (w, h))
                # pre_pro[pre_pro2 >= 0.5] = 170
                # pre_pro[pre_pro2 < 0.5] = 0

                pre_pro = pre_pro.astype("int32")
                # print(pre_pro)
                # pixel = np.unique(pre_pro)
                # print(pixel)

                cv2.imwrite(save_name, pre_pro)
                # pre_img = Image.fromarray(pre_pro)
                # pre_img.save(save_name)

