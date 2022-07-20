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

if __name__ == "__main__":
    net = param.create_model
    # 将网络拷贝到deivce中
    net.to(device=param.device)
    # 加载模型参数
    net.load_state_dict(
        torch.load(param.model_pth, map_location=param.device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = './Heart Data/training'

    # 评价指标初始化
    avg_DSI = 0.
    avg_DSI1 = 0.
    avg_DSI2 = 0.
    avg_DSI3 = 0.

    avg_Precision = 0.
    avg_Precision1 = 0.
    avg_Precision2 = 0.
    avg_Precision3 = 0.

    avg_Recall = 0.
    avg_Recall1 = 0.
    avg_Recall2 = 0.
    avg_Recall3 = 0.

    avg_F1 = 0.
    avg_F1_1 = 0.
    avg_F1_2 = 0.
    avg_F1_3 = 0.

    count = 0
    # 遍历素有图片
    for cls in os.listdir(tests_path):
        # DCM/HCM/NOR
        patients_img = os.path.join(tests_path, cls, "png", "Image")
        patients_lab = os.path.join(tests_path, cls, "png", "Label")
        patients_sav = os.path.join(tests_path, cls, "png", "Result")
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
                img = data_transform(img)
                img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
                img = img.to(device=param.device)

                # 预测
                pre = net(img)
                # 提取结果
                pre = np.array(pre.data.cpu())[0]
                # 处理结果
                pre_pro = np.zeros((h, w), dtype=uint8)
                pre_pro1 = cv2.resize(pre[0], (w, h))
                pre_pro[pre_pro1 >= 0.5] = 85
                pre_pro[pre_pro1 < 0.5] = 0

                pre_pro2 = cv2.resize(pre[1], (w, h))
                pre_pro[pre_pro2 >= 0.5] = 170

                pre_pro3 = cv2.resize(pre[2], (w, h))
                pre_pro[pre_pro3 >= 0.5] = 255

                pre_pro = pre_pro.astype("int32")
                # print(pre_pro)
                # pixel = np.unique(pre_pro)
                # print(pixel)

                # evaluate
                lab = Image.open(lab_path).convert('L')
                lab = np.array(lab)

                # 指标评价
                # DICE
                pre = pre_pro
                DSI1 = calDice(lab, pre, 85)
                avg_DSI1 = avg_DSI1 + DSI1
                print('（1.1） 85_DICE计算结果，      DSI       = {0:.4}'.format(DSI1))
                DSI2 = calDice(lab, pre, 170)
                avg_DSI2 = avg_DSI2 + DSI2
                print('（1.2）170_DICE计算结果，      DSI       = {0:.4}'.format(DSI2))
                DSI3 = calDice(lab, pre, 255)
                avg_DSI3 = avg_DSI3 + DSI3
                print('（1.3）255_DICE计算结果，      DSI       = {0:.4}'.format(DSI3))
                DSI = (DSI1 + DSI2 + DSI3) / 3.
                avg_DSI = avg_DSI + DSI
                print('（ 1 ）AVG_DICE计算结果，      DSI       = {0:.4}'.format(DSI))

                # Precision
                Precision1 = calPrecision(lab, pre, 85)
                avg_Precision1 = avg_Precision1 + Precision1
                print('（2.1） 85_Precision计算结果， Precision = {0:.4}'.format(Precision1))
                Precision2 = calPrecision(lab, pre, 170)
                avg_Precision2 = avg_Precision2 + Precision2
                print('（2.2）170_Precision计算结果， Precision = {0:.4}'.format(Precision2))
                Precision3 = calPrecision(lab, pre, 255)
                avg_Precision3 = avg_Precision3 + Precision3
                print('（2.3）255_Precision计算结果， Precision = {0:.4}'.format(Precision3))
                Precision = (Precision1 + Precision2 + Precision3)/3.
                avg_Precision = avg_Precision + Precision
                print('（ 2 ）AVG_Precision计算结果， Precision = {0:.4}'.format(Precision))

                # Recall
                Recall1 = calRecall(lab, pre, 85)
                avg_Recall1 = avg_Recall1 + Recall1
                print('（3.1） 85_Recall计算结果，    Recall    = {0:.4}'.format(Recall1))
                Recall2 = calRecall(lab, pre, 170)
                avg_Recall2 = avg_Recall2 + Recall2
                print('（3.2）170_Recall计算结果，    Recall    = {0:.4}'.format(Recall2))
                Recall3 = calRecall(lab, pre, 255)
                avg_Recall3 = avg_Recall3 + Recall3
                print('（3.3）255_Recall计算结果，    Recall    = {0:.4}'.format(Recall3))
                Recall = (avg_Recall1 + avg_Recall2 + avg_Recall3)/3.
                avg_Recall = avg_Recall + Recall
                print('（ 3 ）AVG_Recall计算结果，    Recall    = {0:.4}'.format(Recall))

                # F1
                F1_1 = 2. * Precision1 * Recall1 / (Precision1 + Recall1)
                avg_F1_1 += F1_1
                print('（4.1） 85_F1计算结果，         F1       = {0:.4}'.format(F1_1))  # 保留四位有效数字
                F1_2 = 2. * Precision2 * Recall2 / (Precision2 + Recall2)
                avg_F1_2 += F1_2
                print('（4.2）170_F1计算结果，         F1       = {0:.4}'.format(F1_2))  # 保留四位有效数字
                F1_3 = 2. * Precision3 * Recall3 / (Precision3 + Recall3)
                avg_F1_3 += F1_3
                print('（4.3）255_F1计算结果，         F1       = {0:.4}'.format(F1_3))  # 保留四位有效数字
                F1 = (F1_1 + F1_2 + F1_3) / 3.
                avg_F1 += F1
                print('（ 4 ）AVG_F1计算结果，         F1       = {0:.4}'.format(F1))  # 保留四位有效数字

                print("\n")
                count += 1

                cv2.imwrite(save_name, pre_pro)
                # pre_img = Image.fromarray(pre_pro)
                # pre_img.save(save_name)

    # 计算均值
    print('最终取均值结果：')
    avg_DSI1 = avg_DSI1 / count
    print('（1.1） 85_DICE计算结果，      DSI       = {0:.4}'.format(avg_DSI1))
    avg_DSI2 = avg_DSI2 / count
    print('（1.2）170_DICE计算结果，      DSI       = {0:.4}'.format(avg_DSI2))
    avg_DSI3 = avg_DSI3 / count
    print('（1.3）255_DICE计算结果，      DSI       = {0:.4}'.format(avg_DSI3))
    avg_DSI = avg_DSI / count
    print('（ 1 ）AVG_DICE计算结果，      DSI       = {0:.4}'.format(avg_DSI))

    avg_Precision1 = avg_Precision1 / count
    print('（2.1） 85_Precision计算结果， Precision = {0:.4}'.format(avg_Precision1))
    avg_Precision2 = avg_Precision2 / count
    print('（2.2）170_Precision计算结果， Precision = {0:.4}'.format(avg_Precision2))
    avg_Precision3 = avg_Precision3 / count
    print('（2.3）255_Precision计算结果， Precision = {0:.4}'.format(avg_Precision3))
    avg_Precision = avg_Precision / count
    print('（ 2 ）AVG_Precision计算结果， Precision = {0:.4}'.format(avg_Precision))

    avg_Recall1 = avg_Recall1 / count
    print('（3.1） 85_Recall计算结果，    Recall    = {0:.4}'.format(avg_Recall1))
    avg_Recall2 = avg_Recall2 / count
    print('（3.2）170_Recall计算结果，    Recall    = {0:.4}'.format(avg_Recall2))
    avg_Recall3 = avg_Recall3 / count
    print('（3.3）255_Recall计算结果，    Recall    = {0:.4}'.format(avg_Recall3))
    avg_Recall = avg_Recall / count
    print('（ 3 ）AVG_Recall计算结果，    Recall    = {0:.4}'.format(avg_Recall))

    avg_F1_1 = avg_F1_1 / count
    print('（4.1） 85_F1计算结果，        F1        = {0:.4}'.format(avg_F1_1))
    avg_F1_2 = avg_F1_2 / count
    print('（4.2）170_F1计算结果，        F1        = {0:.4}'.format(avg_F1_2))
    avg_F1_3 = avg_F1_3 / count
    print('（4.3）255_F1计算结果，        F1        = {0:.4}'.format(avg_F1_3))
    avg_F1 = avg_F1 / count
    print('（ 4 ）avg_F1计算结果，        F1        = {0:.4}'.format(avg_F1))

