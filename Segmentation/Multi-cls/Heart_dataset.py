import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class HeartDataset(Dataset):
    def __init__(self, root: str, train: bool, ki=0, k_folds=10, typ='train', img_cha=1, transforms=None):
        super(HeartDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "Heart Data", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        self.all_img_list = []
        self.all_label_list = []
        for cls in os.listdir(data_root):
            patients_folders = os.path.join(data_root, cls, "png", "Image")
            for patient in os.listdir(patients_folders):
                if patient != '15':
                    img_folder = os.path.join(patients_folders, patient)
                    lab_folder = os.path.join(data_root, cls, "png", "Label", patient)
                for img in os.listdir(img_folder):
                    self.all_img_list.append(os.path.join(img_folder, img))
                    self.all_label_list.append(os.path.join(lab_folder, img.replace("image", "label")))

        # check files
        for i in self.all_img_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        for i in self.all_label_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # k_folds
        if train:
            length = len(self.all_img_list)
            every_z_len = length // k_folds
            if typ == 'val':
                self.img_list = self.all_img_list[every_z_len * ki: every_z_len * (ki + 1)]
                self.label_list = self.all_label_list[every_z_len * ki: every_z_len * (ki + 1)]
            elif typ == 'train':
                self.img_list = self.all_img_list[: every_z_len * ki] + self.all_img_list[every_z_len * (ki + 1):]
                self.label_list = self.all_label_list[: every_z_len * ki] + self.all_label_list[every_z_len * (ki + 1):]

        # image shape
        self.img_cha = img_cha

    def __getitem__(self, idx):
        if self.img_cha == 1:
            img = Image.open(self.img_list[idx]).convert('L')
        else:
            img = Image.open(self.img_list[idx]).convert('RGB')
        lab = Image.open(self.label_list[idx]).convert('L')
        # encode label
        lab = np.array(lab)
        # pixel = np.unique(lab)
        # print(pixel)
        lab = lab.astype(str)

        lab_pro = np.zeros((3, lab.shape[0], lab.shape[1]))
        idx1 = {'0': 0, '170': 0, '255': 0, '85': 1}
        idx2 = {'0': 0, '170': 1, '255': 0, '85': 0}
        idx3 = {'0': 0, '170': 0, '255': 1, '85': 0}

        ix, jx = lab.shape
        for i in range(ix):
            for j in range(jx):
                pixel = lab[i][j]
                pixelid1 = idx1[pixel]
                lab_pro[0][i][j] = pixelid1
                pixelid2 = idx2[pixel]
                lab_pro[1][i][j] = pixelid2
                pixelid3 = idx3[pixel]
                lab_pro[2][i][j] = pixelid3

        lab_pro = lab_pro.astype("int32")
        # pixel = np.unique(lab_pro)
        # print(pixel)

        # # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        lab0 = Image.fromarray(lab_pro[0])
        lab1 = Image.fromarray(lab_pro[1])
        lab2 = Image.fromarray(lab_pro[2])

        if self.transforms is not None:
            img = self.transforms(img)
            lab0 = self.transforms(lab0)
            lab1 = self.transforms(lab1)
            lab2 = self.transforms(lab2)
            lab_proc = np.zeros((3, lab1.shape[1], lab1.shape[2]))
            lab_proc[0] = lab0
            lab_proc[1] = lab1
            lab_proc[2] = lab2
            lab_proc = lab_proc.astype("int32")
            # pixel = np.unique(lab_proc)
            # print(pixel)

        return img, lab_proc

    def __len__(self):
        return len(self.img_list)


class MultiHeartDataset(Dataset):
    def __init__(self, root: str, train: bool, ki=0, k_folds=10, typ='train', img_cha=1, transforms=None):
        super(MultiHeartDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "Heart Data", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        self.all_img_list = []
        self.all_label_list = []
        for cls in os.listdir(data_root):
            patients_folders = os.path.join(data_root, cls, "png", "Image")
            for patient in os.listdir(patients_folders):
                if patient != '15':  # 15--20
                    img_folder = os.path.join(patients_folders, patient)
                    lab_folder = os.path.join(data_root, cls, "png", "Label", patient)
                for img in os.listdir(img_folder):
                    self.all_img_list.append(os.path.join(img_folder, img))
                    self.all_label_list.append(os.path.join(lab_folder, img.replace("image", "label")))

        # check files
        for i in self.all_img_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        for i in self.all_label_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # k_folds
        if train:
            length = len(self.all_img_list)
            every_z_len = length // k_folds
            if typ == 'val':
                self.img_list = self.all_img_list[every_z_len * ki: every_z_len * (ki + 1)]
                self.label_list = self.all_label_list[every_z_len * ki: every_z_len * (ki + 1)]
            elif typ == 'train':
                self.img_list = self.all_img_list[: every_z_len * ki] + self.all_img_list[every_z_len * (ki + 1):]
                self.label_list = self.all_label_list[: every_z_len * ki] + self.all_label_list[every_z_len * (ki + 1):]

        # image shape
        self.img_cha = img_cha

    def __getitem__(self, idx):
        if self.img_cha == 1:
            img = Image.open(self.img_list[idx]).convert('L')
        else:
            img = Image.open(self.img_list[idx]).convert('RGB')
        lab = Image.open(self.label_list[idx]).convert('L')
        # encode label
        lab = np.array(lab)
        # pixel = np.unique(lab)
        # print(pixel)
        lab = lab.astype(str)

        lab_pro = np.zeros((3, lab.shape[0], lab.shape[1]))
        idx1 = {'0': 0, '170': 0, '255': 0, '85': 1}
        idx2 = {'0': 0, '170': 1, '255': 0, '85': 0}
        idx3 = {'0': 0, '170': 0, '255': 1, '85': 0}

        ix, jx = lab.shape
        for i in range(ix):
            for j in range(jx):
                pixel = lab[i][j]
                pixelid1 = idx1[pixel]
                lab_pro[0][i][j] = pixelid1
                pixelid2 = idx2[pixel]
                lab_pro[1][i][j] = pixelid2
                pixelid3 = idx3[pixel]
                lab_pro[2][i][j] = pixelid3

        lab_pro = lab_pro.astype("int32")
        # pixel = np.unique(lab_pro)
        # print(pixel)

        # # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        lab0 = Image.fromarray(lab_pro[0])
        lab1 = Image.fromarray(lab_pro[1])
        lab2 = Image.fromarray(lab_pro[2])

        if self.transforms is not None:
            img = self.transforms(img)
            lab0 = self.transforms(lab0)
            lab1 = self.transforms(lab1)
            lab2 = self.transforms(lab2)
            lab_proc = np.zeros((1, lab1.shape[1], lab1.shape[2]))
            # lab_proc = np.zeros((2, lab1.shape[1], lab1.shape[2]))
            lab_proc[0] = lab1
            # lab_proc[0] = lab0
            # lab_proc[1] = lab2
            lab_proc = lab_proc.astype("int32")
            # pixel = np.unique(lab_proc)
            # print(pixel)

        return img, lab_proc

    def __len__(self):
        return len(self.img_list)


class SmallHeartDataset(Dataset):
    def __init__(self, root: str, train: bool, ki=0, k_folds=10, typ='train', img_cha=1, transforms=None):
        super(SmallHeartDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "Heart Data", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        self.all_img_list = []
        self.all_label_list = []
        for cls in os.listdir(data_root):
            patients_folders = os.path.join(data_root, cls, "png", "Image")
            for patient in os.listdir(patients_folders):
                if patient != '15':
                    img_folder = os.path.join(patients_folders, patient)
                    lab_folder = os.path.join(data_root, cls, "png", "Label", patient)
                imgs = os.listdir(img_folder)
                small_img = imgs[-1:]
                for img in small_img:
                    self.all_img_list.append(os.path.join(img_folder, img))
                    self.all_label_list.append(os.path.join(lab_folder, img.replace("image", "label")))

        # check files
        for i in self.all_img_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        for i in self.all_label_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # k_folds
        if train:
            length = len(self.all_img_list)
            every_z_len = length // k_folds
            if typ == 'val':
                self.img_list = self.all_img_list[every_z_len * ki: every_z_len * (ki + 1)]
                self.label_list = self.all_label_list[every_z_len * ki: every_z_len * (ki + 1)]
            elif typ == 'train':
                self.img_list = self.all_img_list[: every_z_len * ki] + self.all_img_list[every_z_len * (ki + 1):]
                self.label_list = self.all_label_list[: every_z_len * ki] + self.all_label_list[every_z_len * (ki + 1):]

        # image shape
        self.img_cha = img_cha

    def __getitem__(self, idx):
        if self.img_cha == 1:
            img = Image.open(self.img_list[idx]).convert('L')
        else:
            img = Image.open(self.img_list[idx]).convert('RGB')
        lab = Image.open(self.label_list[idx]).convert('L')
        # encode label
        lab = np.array(lab)
        # pixel = np.unique(lab)
        # print(pixel)
        lab = lab.astype(str)

        lab_pro = np.zeros((3, lab.shape[0], lab.shape[1]))
        idx1 = {'0': 0, '170': 0, '255': 0, '85': 1}
        idx2 = {'0': 0, '170': 1, '255': 0, '85': 0}
        idx3 = {'0': 0, '170': 0, '255': 1, '85': 0}

        ix, jx = lab.shape
        for i in range(ix):
            for j in range(jx):
                pixel = lab[i][j]
                pixelid1 = idx1[pixel]
                lab_pro[0][i][j] = pixelid1
                pixelid2 = idx2[pixel]
                lab_pro[1][i][j] = pixelid2
                pixelid3 = idx3[pixel]
                lab_pro[2][i][j] = pixelid3

        lab_pro = lab_pro.astype("int32")
        # pixel = np.unique(lab_pro)
        # print(pixel)

        # # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        lab0 = Image.fromarray(lab_pro[0])
        lab1 = Image.fromarray(lab_pro[1])
        lab2 = Image.fromarray(lab_pro[2])

        if self.transforms is not None:
            img = self.transforms(img)
            lab0 = self.transforms(lab0)
            lab1 = self.transforms(lab1)
            lab2 = self.transforms(lab2)
            lab_proc = np.zeros((3, lab1.shape[1], lab1.shape[2]))
            lab_proc[0] = lab0
            lab_proc[1] = lab1
            lab_proc[2] = lab2
            lab_proc = lab_proc.astype("int32")
            # pixel = np.unique(lab_proc)
            # print(pixel)

        return img, lab_proc

    def __len__(self):
        return len(self.img_list)