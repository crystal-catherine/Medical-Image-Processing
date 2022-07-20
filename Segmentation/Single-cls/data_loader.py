import os
import cv2
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Data_Loader(Dataset):
    def __init__(self, data_path, ki=0, K=5, typ='train', rand=False, img_size=512, imgcha=1, transform=None):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.all_imgs_path = glob.glob(os.path.join(self.data_path, 'images/*.tif'))
        self.img_size = img_size
        self.img_cha = imgcha
        self.transform = transform

        if rand:
            random.seed(1)
            random.shuffle(self.all_imgs_path)

        length = len(self.all_imgs_path)
        every_z_len = length // K
        if typ == 'val':
            self.imgs_path = self.all_imgs_path[every_z_len * ki: every_z_len * (ki + 1)]
        elif typ == 'train':
            self.imgs_path = self.all_imgs_path[: every_z_len * ki] + self.all_imgs_path[every_z_len * (ki + 1):]

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        label_name = image_path.replace('images', '1st_manual')
        label_name = label_name.replace('training', 'manual1')
        label_path = label_name.replace('.tif', '.gif')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = Image.open(label_path)
        label = label.resize((self.img_size, self.img_size))
        label = cv2.cvtColor(np.array(label), cv2.COLOR_RGB2BGR)
        # 将标签转为单/绿色通道的图片
        if self.img_cha == 1:
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            b, image, r = cv2.split(image)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # resize
        if self.img_size > image.shape[1]:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
        # 标签二值化
        t, label = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY)
        # 转化为张量
        if self.transform is None:
            image = image.reshape(self.img_cha, image.shape[0], image.shape[1])
        else:
            image = self.transform(image)

        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)
