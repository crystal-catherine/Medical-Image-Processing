from PIL import Image
from PIL import ImageEnhance
from PIL import ImageChops
import os
import numpy as np

# 数据增强

# 1、平移。在图像平面上对图像以一定方式进行平移。
# 2、翻转图像。沿着水平或者垂直方向翻转图像。
# 3、旋转角度。随机旋转图像一定角度; 改变图像内容的朝向。
# 4、随机颜色。包括调整图像饱和度、亮度、对比度、锐度
# 5、添加噪声


# 1、图像平移
def move(img, lab):  # 水平位移，垂直位移
    x_off = np.random.randint(1, 20)
    y_off = np.random.randint(1, 40)
    img_offset = ImageChops.offset(img, x_off, y_off)
    lab_offset = ImageChops.offset(lab, x_off, y_off)
    return img_offset, lab_offset


# 2、翻转图像
def flip(img, lab):
    factor = np.random.randint(1, 3)  # 随机因子，随机上下或者左右翻转
    if factor == 1:
        filp_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        filp_lab = lab.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        filp_lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
    return filp_img, filp_lab


#  3、旋转角度
def rotation(img, lab):
    factor = np.random.randint(1, 21)  # 随机旋转角度
    rotation_img = img.rotate(factor)
    rotation_lab = lab.rotate(factor)
    return rotation_img, rotation_lab


# 4、随机颜色
def color(img, lab):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(5, 15) / 10.  # 随机因子
    color_image = ImageEnhance.Color(img).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(8, 15) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 13) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(5, 31) / 10.  # 随机因子
    random_color = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
    return random_color, lab


# 5、随机添加黑白噪声
def salt_and_pepper_noise(img, lab, proportion=0.00025):
    noise_img = img
    height, width = noise_img.size[0], noise_img.size[1]
    proportion = proportion * np.random.randint(1, 50)
    num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
    img_pixels = noise_img.load()
    for i in range(num):
        w = np.random.randint(0, width - 1)
        h = np.random.randint(0, height - 1)
        if np.random.randint(0, 2) == 1:
            img_pixels[h, w] = 0
        else:
            img_pixels[h, w] = 255
    return noise_img, lab


def Gaussian_noise(img, lab, mean=0, std=0.05):
    image = np.array(img).copy()
    image = image/255.0
    noise = np.random.normal(mean, std ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    out = Image.fromarray(out).convert("RGB")
    return out, lab


# 概率执行函数
def random_run(probability, func, useimage, uselab):
    """以probability%的概率执行func(*args)"""
    list = []
    for i in range(probability):
        list.append(1)  # list中放入probability个1
    for x in range(100 - probability):
        list.append(0)  # 剩下的位置放入0
    a = np.random.choice(list)  # 随机抽取一个
    if a == 0:
        return useimage, uselab
    if a == 1:
        image, label = func(useimage, uselab)
        return image, label


def Augumentation(imageDir, saveDir):
    seed = 50  # 每张初始图片要数据增强为多少张图片
    for name in os.listdir(imageDir):
        i = 0
        for i in range(seed):
            i = i + 1
            # 图片
            img_saveName = str(name[:-4]) + "_" + str(i) + ".tif"

            imgName = os.path.join(imageDir, name)
            print(imgName)
            img = Image.open(imgName)

            # 标签
            lab_saveName = img_saveName.replace('training', 'manual1')
            lab_saveName = lab_saveName.replace('.tif', '.gif')

            labName = imgName.replace('images', '1st_manual')
            labName = labName.replace('training', 'manual1')
            labName = labName.replace('.tif', '.gif')
            print(labName)
            lab = Image.open(labName)

            saveImage, saveLabel = random_run(60, flip, img, lab)  # 翻转
            saveImage, saveLabel = random_run(70, color, saveImage, saveLabel)  # 色彩变化
            saveImage, saveLabel = random_run(30, move, saveImage, saveLabel)  # 平移
            saveImage, saveLabel = random_run(50, rotation, saveImage, saveLabel)  # 旋转
            saveImage, saveLabel = random_run(5, salt_and_pepper_noise, saveImage, saveLabel)  # 添加黑白噪声点
            saveImage, saveLabel = random_run(1, Gaussian_noise, saveImage, saveLabel)  # 添加高斯噪声点

            if saveImage != None:
                saveImage.save(os.path.join(saveDir, img_saveName))
                saveDir2 = saveDir.replace('images', '1st_manual')
                if not os.path.exists(saveDir2):
                    os.makedirs(saveDir2)
                saveLabel.save(os.path.join(saveDir2, lab_saveName))
            else:
                pass
            print(i)


if __name__ == "__main__":
    imageDir = "./Dataset/train/images"  # 要改变的图片的路径文件夹
    saveDir = "./Dataset/process_train/images/"  # 要保存的图片的路径文件夹
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    Augumentation(imageDir, saveDir)