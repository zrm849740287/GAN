import glob
import os
import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

root = 'E:\ZRM\data\\nmi_pair'
#mode = 'trainA'

# print(os.path.join(root,'trainA') + "/*.*") # 因为这里不对 没有去我们指定的目录 datasets读数据部分就得按照我们的数据情况改


files = sorted(glob.glob(os.path.join(root, 'trainA') + "/*.*"))
# print(sorted(glob.glob(os.path.join(root, 'trainA') + "/*.*")))
# img = Image.open(files[index % len(files)])
#         w, h = img.size
#         img_A = img.crop((0, 0, w / 2, h))
#         img_B = img.crop((w / 2, 0, w, h))

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)

        self.files1 = sorted(glob.glob(os.path.join(root,"trainA") + "/*.*"))
        self.files2 = sorted(glob.glob(os.path.join(root,'trainB') + "/*.*"))

    def __getitem__(self, index):
        img_A = Image.open(self.files1[index % len(self.files)])
        img_B = Image.open(self.files1[index % len(self.files)])
        print(img_A)

        #w, h = img.size
        # img_A = imgA.crop((0, 0, w / 2, h))
        # img_B = imgB.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A), "RGB")
            img_B = Image.fromarray(np.array(img_B), "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)
print(Dataset)