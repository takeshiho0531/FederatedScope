import glob
import os.path as osp
import torch.utils.data as data
from torchvision import models, transforms
from PIL import Image
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models

from tqdm import tqdm
from federatedscope.register import register_data

def load_my_data(config, client_cfgs=None):
    from federatedscope.core.data import BaseDataTranslator
    # Load a dataset, whose class is `torch.utils.data.Dataset`
    train_dataset = InfantDataset(
        file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
    val_dataset = InfantDataset(
        file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')
		# Instantiate a translator according to config
    translator = BaseDataTranslator(config, client_cfgs)
    # Translate torch dataset to FS data
    fs_data = translator([train_dataset, [], val_dataset])
    return fs_data, config

def call_my_data(config):
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
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)),  # データオーギュメンテーション
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

def make_datapath_list(phase="train"):
    """
    データのパスを格納したリストを作成する。

    Parameters
    ----------
    phase : 'train' or 'val'
        訓練データか検証データかを指定する

    Returns
    -------
    path_list : list
        データへのパスを格納したリスト
    """

    rootpath = "/content/drive/MyDrive/NakaoLab/donateacry-corpus-master/donateacry-android-upload-bucket-jpg-copy/"
    target_path = osp.join(rootpath+phase+'/*.jpg')
    print(target_path)

    path_list = []  # ここに格納する

    # globを利用してサブディレクトリまでファイルパスを取得する
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list

class InfantDataset(data.Dataset):
    """
    Datasetクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])

        # 画像のラベルをファイル名から抜き出す
        if self.phase == "train":
            label = img_path[-6:-4]
        elif self.phase == "val":
            label = img_path[-6:-4]

        # ラベルを数値に変更する
        if label == "hu":
            label = 0
        elif label == "bu":
            label = 1
        elif label == "bp":
            label = 1
        elif label == "dc":
            label = 1
        elif label == "ti":
            label = 1

        return img_transformed, label

train_list = make_datapath_list(phase="train")
val_list = make_datapath_list(phase="val")

# Datasetを作成する
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


