import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import gloDataset
import os
import  time

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

def train_model(model, criterion, optimizer, dataload, num_epochs=5):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
        time.sleep(30)
    torch.save(model.state_dict(), 'C:\\RESULT\\'+'weights_%d.pth' % epoch)
    return model

#训练模型
def train(args):
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    glo_dataset = gloDataset("C:/traindataset/",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(glo_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)

#显示模型的输出结果
def test(args):
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    glo_dataset = gloDataset("c:/traindataset/", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(glo_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    from skimage import io
    plt.ion()
    i = 0
    glo_path = glo_dataset.imgs[0][0]
    mask_path = glo_path[:glo_path.find('_')-3] + 'mask'
    mkdir(mask_path)
    with torch.no_grad():
        for x, _ in dataloaders:
            img_list = glo_dataset.imgs
            img_name = img_list[i][0]
            mask_rst = img_name[img_name.find('_')+7:-4]
            y=model(x).sigmoid()
            img_y=torch.squeeze(y).numpy()
            io.imsave(mask_path + '/' + mask_rst + '.png',img_y)
            i = i + 1


if __name__ == '__main__':
    #参数解析
    parse = argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=25)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)