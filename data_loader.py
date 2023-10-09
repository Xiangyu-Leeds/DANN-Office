import torch.utils.data as data
from PIL import Image
import os


class GetLoader(data.Dataset):
    def __init__(self, data_root, transform=None):
        self.root = data_root
        self.transform = transform

        # f = open(data_list, 'r')
        # data_list = f.readlines()
        # f.close()

        # self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []
        # print(self.root)

        dirs = sorted(os.listdir(data_root))
        # print(dirs)
        for label, category in enumerate(dirs):
            category_path = os.path.join(data_root, category)
            # 遍历每一个子目录下的所有图片
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                self.img_paths.append(img_path)
                self.img_labels.append(label)

        # for data in data_list:
        #     self.img_paths.append(data[:-3])
        #     self.img_labels.append(data[-2])

        # for img_file in os.listdir(data_root):
        #     if img_file.endswith('.png'):  # 假设图像文件都是以 .jpg 结尾
        #         self.img_paths.append(img_file)
                # print(self.img_paths)
                # self.img_labels.append(int(img_file[-5]))  # 假设标签信息位于文件名的倒数第五个字符

        # self.n_data = len(self.img_paths)

    def __getitem__(self, item):
        img_paths, label = self.img_paths[item], self.img_labels[item]
        img = Image.open(img_paths).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            # labels = int(labels)

        return img, label

    def __len__(self):
        return len(self.img_paths)
