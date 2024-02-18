import cv2
import tqdm
import torch
import numpy as np
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset


# def readSingleImage(pth, target_shape=(109, 84)):
def readSingleImage(pth, target_shape=(32, 32)):
    """
    输入路径pth，输出这个图形的ndarray形式，并将其调整为目标形状
    :param pth: 图形路径
    :param target_shape: 目标形状，例如 (desired_height, desired_width)
    :return: 调整后的图形的ndarray形式
    """
    img = cv2.imread(pth)

    # 调整图像大小
    resized_img = cv2.resize(img, target_shape)

    # 将图像转换为PyTorch的Tensor
    tensor_img = torch.from_numpy(np.transpose(resized_img, (2, 0, 1))).float()

    return tensor_img


class PicDataset(Dataset):
    """
    编写Dataset，需要重写如下三个方法
    对于图像类数据，建议仅将路径存放在类中，然后使用图像读取函数读取图像作为数据即可
    """
    def __init__(self, PngPath, _label):
        super().__init__()
        self.picLst = [os.path.join(PngPath, I) for I in os.listdir(PngPath)]
        self.label = _label


    def __getitem__(self, item):
        return readSingleImage(self.picLst[item]), self.label
    # 返回样本的时候返回标签和图形，标签就在这里打

    def __len__(self):
        return len(self.picLst)


if __name__ == '__main__':
    TrainingFemale = PicDataset(
        '/Users/andrewlee/Desktop/Projects/动手深度学习/实操/CV_Project/archive/Training/female',
        0
    )
    TrainingMale = PicDataset(
        '/Users/andrewlee/Desktop/Projects/动手深度学习/实操/CV_Project/archive/Training/male',
        1
    )
    female_loader = DataLoader(TrainingFemale + TrainingMale, 5, True)
    for data, label in female_loader:
        print(data.shape, label)
    print("end")