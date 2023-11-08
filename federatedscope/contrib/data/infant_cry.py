import glob
import os
import os.path as osp
import torch.utils.data as data
from torchvision import models, transforms
from PIL import Image
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from federatedscope.register import register_data
from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed


def load_my_data(config, client_cfgs=None):
    from federatedscope.core.data import DummyDataTranslator

    splits = config.data.splits
    path = config.data.root

    setup_seed(12345)
    dataset = InfantDataset(root=path,
                            s_frac=config.data.subsample,
                            tr_frac=splits[0],
                            val_frac=splits[1],
                            transform=ImageTransform(size, mean, std))
    print("len(dataset)", len(dataset))
    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    print("client_num", client_num)
    config.merge_from_list(['federate.client_num', client_num])
    # Convert list to dict
    data_dict = dict()
    for client_idx in range(1, client_num + 1):
        data_dict[client_idx] = dataset[client_idx - 1]
    translator = DummyDataTranslator(config, client_cfgs)
    data = translator(data_dict)
    data = convert_data_mode(data, config)
    setup_seed(config.seed)
    return data, config


def call_my_data(config, client_cfgs=None):
    if config.data.type == "mydata":
        data, modified_config = load_my_data(config)
        return data, modified_config


register_data("mydata", call_my_data)


class ImageTransform():
    """
    画像の前処理クラス。訓練時、検証時で異なる動作をする。
    画像のサイズをリサイズし、色を標準化する。
    訓練時はRandomResizedCropとRandomHorizontalFlipでデータオーギュメンテーションする。

    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ。
    mean : (R, G, B)
        各色チャネルの平均値。
    std : (R, G, B)
        各色チャネルの標準偏差。
    """
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize,
                                             scale=(0.5,
                                                    1.0)),  # データオーギュメンテーション
                transforms.RandomHorizontalFlip(),  # データオーギュメンテーション
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img)


class InfantDataset(data.Dataset):
    """
    Datasetクラス。PyTorchのDatasetクラスを継承。

    Arguments:
        root (str): root path.
        name (str): name of dataset
        s_frac (float): fraction of the dataset to be used; default=0.3.
        tr_frac (float): train set proportion for each task; default=0.8.
        val_frac (float): valid set proportion for each task; default=0.0.
        train_tasks_frac (float): fraction of test tasks; default=1.0.
        transform: transform for x.
        target_transform: transform for y.
    """
    def __init__(self,
                 root,
                 s_frac=0.3,
                 tr_frac=0.8,
                 val_frac=0.0,
                 train_tasks_frac=1.0,
                 seed=123,
                 transform=None,
                 target_transform=None):
        self.s_frac = s_frac
        self.tr_frac = tr_frac
        self.val_frac = val_frac
        self.seed = seed
        self.train_tasks_frac = train_tasks_frac  # ??
        self.transform = transform
        self.root = "/FederatedScope/data/example/"

    def __len__(self):
        return len([f.path for f in os.scandir(self.root) if f.is_dir()])

    def __getitem__(self, index):
        """
        Arguments:
            index (int): Index

        :returns:
            dict: {'train':[(image, target)],
                   'test':[(image, target)],
                   'val':[(image, target)]}
            where target is the target class.
        """

        # index番目の画像をロード
        train_img_path = glob.glob(
            os.path.join(self.root + str(index) + "/train", '*.jpg'))
        test_img_path = glob.glob(
            os.path.join(self.root + str(index) + "/test", '*.jpg'))
        val_img_path = glob.glob(
            os.path.join(self.root + str(index) + "/val", '*.jpg'))
        print("train_img_path", train_img_path)
        print(self.root + str(index) + "train")
        train_img = Image.open(train_img_path[0])  # [高さ][幅][色RGB]
        test_img = Image.open(test_img_path[0])
        val_img = Image.open(val_img_path[0])

        # 画像の前処理を実施
        train_img_transformed = self.transform(
            train_img, "train")  # torch.Size([3, 224, 224])
        test_img_transformed = self.transform(test_img, "val")
        val_img_transformed = self.transform(val_img, "val")

        # 画像のラベルをファイル名から抜き出す
        train_label = convert_label(train_img_path[-6:-4])
        test_label = convert_label(test_img_path[-6:-4])
        val_label = convert_label(val_img_path[-6:-4])

        img_dict = {}
        img_dict["train"] = [(train_img_transformed, train_label)]
        img_dict["test"] = [(test_img_transformed, test_label)]
        img_dict["val"] = [(val_img_transformed, val_label)]
        return img_dict


def convert_label(label):
    if label == "hu":
        label = 0
    elif label == "bu":
        label = 1
    elif label == "bp":
        label = 2
    elif label == "dc":
        label = 3
    elif label == "ti":
        label = 4
    else:
        label = 5
    return label


# Datasetを作成する
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
