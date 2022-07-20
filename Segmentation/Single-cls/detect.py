import os
import cv2
import glob
import numpy as np
from torchvision import transforms
from PIL import Image
from nets import TransUnet
from score import *


if __name__ == "__main__":
    img_size = 560
    blocks = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TransUnet(in_channels=1, img_dim=img_size, vit_blocks=blocks, vit_dim_linear_mhsa_block=img_size, classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('model/nor560_TransUNet12b_green_Dice_model_Adam_cos_epoch20.pth', map_location=device))
    net.eval()
    root = "../../test/images"
    img_name_list = os.listdir(root)
    for img_name in img_name_list:
        img_path = os.path.join(root, img_name)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        # green channel
        b, img, r = cv2.split(img)
        img = cv2.resize(img, (img_size, img_size))

        # # no transform
        # img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # img_tensor = torch.from_numpy(img)
        # normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])
        img_tensor = data_transform(img)
        img_tensor = img_tensor.reshape(1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])

        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        # 预测
        seg_img = net(img_tensor)
        # 提取结果
        seg_img = np.array(seg_img.data.cpu()[0])[0]
        # 处理结果
        seg_img[seg_img >= 0.5] = 255
        seg_img[seg_img < 0.5] = 0
        seg_img = cv2.resize(seg_img, (w, h))

        pil_img = Image.fromarray(seg_img)
        pil_img = pil_img.convert("1")
        if not os.path.exists("./pre"):
            os.makedirs("pre")
        pil_img.save(os.path.join("./pre", img_name))
