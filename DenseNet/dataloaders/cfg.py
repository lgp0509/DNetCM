# -*- coding:utf-8 -*-


##数据集的类别
NUM_CLASSES = 2

#数据集的存放位置
DATASET_DIR       = r'c:/DenseNet-with-pytorch-main/data'
TRAIN_DATASET_DIR = r'c:/DenseNet-with-pytorch-main/data/train'
VAL_DATASET_DIR   = r'c:/DenseNet-with-pytorch-main/data/val'
TEST_DATASET_DIR  =r'c:/DenseNet-with-pytorch-main/inference/image'

# DATASET_DIR        = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data'
# TRAIN_DATASET_DIR  = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data/train'
# VAL_DATASET_DIR    = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data/val'
# TEST_DATASET_DIR   = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/inference/image'


#这里需要加入自己的最终预测对应字典，例如:'0': '花'
labels_to_classes = {
    '0' : 'normal',
    '1' : 'sc'
}
