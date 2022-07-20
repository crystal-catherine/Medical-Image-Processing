import os
import cv2
import glob
import numpy as np
from torchvision import transforms
from PIL import Image
from nets import U_Net, AttU_Net, R2U_Net, R2AttU_Net, Upp, TransUnet, MedT
from score import *


if __name__ == "__main__":
    img_size = 560
    TransUNet_vit_block = 12
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    # net = U_Net(img_ch=1, output_ch=1)
    # net = Upp(n_channels=1, n_classes=1)
    # net = AttU_Net(img_ch=1, output_ch=1)
    # net = R2AttU_Net(img_ch=1, output_ch=1)
    net = TransUnet(in_channels=1, img_dim=img_size, vit_blocks=TransUNet_vit_block, vit_dim_linear_mhsa_block=img_size, classes=1)
    # net = MedT(img_size=128, imgchan=1, num_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('model/nor560_TransUNet12b_green_Dice_model_Adam_cos_epoch20.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_folders = '../test/'

    # 评价指标初始化
    avg_DSI = 0.
    avg_Precision = 0.
    avg_Recall = 0.
    avg_F1 = 0.

    count = 0
    # 遍历素有图片
    tests_path = glob.glob(os.path.join(tests_folders, 'images/*.tif'))
    for test_path in tests_path:
        print(count+1)
        img = cv2.imread(test_path)
        print(img.shape)
        h, w, _ = img.shape
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b, img, r = cv2.split(img)
        img = cv2.resize(img, (img_size, img_size))
        # # no transform
        # img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # img_tensor = torch.from_numpy(img)
        # has transform
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])
        img_tensor = data_transform(img)
        img_tensor = img_tensor.reshape(1, 1, img_tensor.shape[1], img_tensor.shape[2])
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        # 预测
        pre = net(img_tensor)
        # 提取结果
        pre = np.array(pre.data.cpu()[0])[0]
        # 处理结果
        pre[pre >= 0.5] = 255
        pre[pre < 0.5] = 0
        pre = cv2.resize(pre, (w, h))

        # 标签
        label_name = test_path.replace('images', '1st_manual')
        # test2 folder
        label_name = label_name.replace('_test', '_manual1')
        # test folder
        # label_name = label_name.replace('training', 'manual1')
        label_path = label_name.replace('.tif', '.gif')
        label = Image.open(label_path)
        # lab = label.resize((512, 512))
        lab = cv2.cvtColor(np.array(label), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
        t, lab = cv2.threshold(lab, 0, 255, cv2.THRESH_BINARY)

        # 指标评价
        DSI = calDice(lab, pre)
        avg_DSI = avg_DSI + DSI
        print('（1）DICE计算结果，      DSI       = {0:.4}'.format(DSI))  # 保留四位有效数字

        Precision = calPrecision(lab, pre)
        avg_Precision = avg_Precision + Precision
        print('（2）Precision计算结果， Precision = {0:.4}'.format(Precision))

        Recall = calRecall(lab, pre)
        avg_Recall = avg_Recall + Recall
        print('（3）Recall计算结果，    Recall    = {0:.4}'.format(Recall))

        F1 = 2.0 * Precision * Recall / (Precision + Recall)
        avg_F1 += F1
        print('（4）F1计算结果，         F1       = {0:.4}'.format(F1))  # 保留四位有效数字

        print("\n")


        # 保存结果地址
        save_res_path = label_path.replace('1st_manual', 'result')
        save_name = save_res_path.split('.')[0] + '.png'
        save_name = save_name.replace('_manual1', '')
        # save_res_path = save_res_path.split('.')[0] + '_pred.png'

        # 保存图片
        cv2.imwrite(save_name, pre)

        count = count + 1

    # 计算均值
    avg_DSI = avg_DSI / count
    avg_Precision = avg_Precision / count
    avg_Recall = avg_Recall / count
    avg_F1 = avg_F1 / count

    print('最终取均值结果：')
    print('（1）DICE计算结果，      DSI       = {0:.4}'.format(avg_DSI))  # 保留四位有效数字
    print('（2）Precision计算结果， Precision = {0:.4}'.format(avg_Precision))
    print('（3）Recall计算结果，    Recall    = {0:.4}'.format(avg_Recall))
    print('（4）F1计算结果，        F1        = {0:.4}'.format(avg_F1))
