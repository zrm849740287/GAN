import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


# class ImageDataset(Dataset):
#     def __init__(self, root, transforms_=None, mode="train"):
#         self.transform = transforms.Compose(transforms_)
#
#         self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
#
#     def __getitem__(self, index):
#
#         img = Image.open(self.files[index % len(self.files)])
#         w, h = img.size
#         img_A = img.crop((0, 0, w / 2, h))
#         img_B = img.crop((w / 2, 0, w, h))
#
#         if np.random.random() < 0.5:
#             img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
#             img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
#
#         img_A = self.transform(img_A)
#         img_B = self.transform(img_B)
#
#         return {"A": img_A, "B": img_B}
#
#     def __len__(self):
#         return len(self.files)
# class ImageDataset(Dataset):
#     def __init__(self, root, transforms_=None):
#         self.transform = transforms.Compose(transforms_)
#
#         self.files1 = sorted(glob.glob(os.path.join(root,"trainA") + "/*.*"))
#         self.files2 = sorted(glob.glob(os.path.join(root,'trainB') + "/*.*"))
#
#     def __getitem__(self, index):
#         imgs_A = Image.open(self.files1[index % len(self.files1)])
#         imgs_B = Image.open(self.files2[index % len(self.files2)])
#
#         # w, h = img.size
#         # img_A = imgA.crop((0, 0, w / 2, h))
#         # img_B = imgB.crop((w / 2, 0, w, h))
#
#         #if np.random.random() < 0.5:
#         imgs_A = Image.fromarray(np.array(imgs_A), "RGB")
#         imgs_B = Image.fromarray(np.array(imgs_B), "RGB")
#
#         imgs_A = self.transform(imgs_A)
#         imgs_B = self.transform(imgs_B)
#
#         return {"A": imgs_A, "B": imgs_B}
#
#     def __len__(self):
#         return len(self.files2)
class ImageDataset(Dataset):
    def __init__(self, root='E:\data\zrm_dataset\\nmi_pair\\train', transforms_=transforms, unaligned=False,
                 mode="train"):  ## (root = "./datasets/facades", unaligned=True:非对其数据)
        self.transform = transforms.Compose(transforms_)  ## transform变为tensor数据
        self.unaligned = unaligned

        self.files_A = glob.glob(os.path.join(root, "%sA" % mode) + "/*.*")  ## "./datasets/facades/trainA/*.*"
        self.files_B = glob.glob(os.path.join(root, "%sB" % mode) + "/*.*")  ## "./datasets/facades/trainB/*.*"

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])  ## 在A中取一张照片
        #
        # if self.unaligned:  ## 如果采用非配对数据，在B中随机取一张
        #     image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        # else:
        image_B = Image.open(self.files_B[index % len(self.files_B)])

        # # 如果是灰度图，把灰度图转换为RGB图
        # if image_A.mode != "RGB":
        #     image_A = to_rgb(image_A)
        # if image_B.mode != "RGB":
        #     image_B = to_rgb(image_B)

        # 把RGB图像转换为tensor图, 方便计算，返回字典数据
        imgs_A = self.transform(image_A)
        imgs_B = self.transform(image_B)
        return {"A": imgs_A, "B": imgs_B}

    ## 获取A,B数据的长度
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))