import torch
import argparse
import param
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim
from torchvision import transforms
from Heart_dataset import *
from nets import init_weights
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim.lr_scheduler as lr_scheduler


def train_net(net, device, args):
    best_loss_list = []
    best_accuracy_list = []
    best_epochs = []

    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    best_epoch = 0

    # 定义优化算法
    opt_type = param.opt_type
    t_max = 25
    params_to_optimize = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=0.9, weight_decay=5e-5)
    optimizer = optim.Adam(params_to_optimize, lr=args.lr)
    # optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.lf)
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 定义Loss算法
    criterion = param.criterion

    data_transform = transforms.Compose([transforms.Resize((param.img_size, param.img_size)),
                                         transforms.ToTensor()])

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 1])  # 8-->4

    for ki in range(args.K):
        print("%d Fold:" % (ki + 1))
        # 加载训练集
        train_dataset = SmallHeartDataset(args.data_path, train=True, ki=ki, k_folds=args.K, typ='train',
                                          img_cha=param.img_cha, transforms=data_transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   )
        val_dataset = SmallHeartDataset(args.data_path, train=True, ki=ki, k_folds=args.K, typ='val',
                                        img_cha=param.img_cha, transforms=data_transform)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=False)
        # 绘图相关
        loss_list = []
        train_loss_list = []
        train_accuracy_list = []
        epoch_list = []
        # 训练epochs次
        for epoch in range(args.epochs):
            print("epoch:" + str(epoch + 1))
            epoch_list.append(epoch + 1)
            # 训练模式
            net.train()
            # 按照batch_size开始训练
            count = 0
            loss_sum = 0.
            for image, label in tqdm(train_loader):
                optimizer.zero_grad()
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # 使用网络参数，输出预测结果
                pred = net(image)

                # 计算loss
                loss = criterion(pred, label)
                loss_list.append(loss)
                loss_sum += loss.item()

                # 更新参数
                loss.backward()
                optimizer.step()
                count = count + 1
            print(str(scheduler.get_last_lr()))

            scheduler.step()
            epoch_loss = loss_sum / count
            print('Loss/train:' + str(epoch_loss))
            epoch_accuracy = 1.0 - epoch_loss
            train_loss_list.append(epoch_loss)
            train_accuracy_list.append(epoch_accuracy)

            if (epoch + 1) % args.val_epoch == 0:
                net.eval()
                with torch.no_grad():
                    num = 1
                    loss = 0.
                    for image, label in val_loader:
                        image = image.to(device=device, dtype=torch.float32)
                        label = label.to(device=device, dtype=torch.float32)
                        pred = net(image)

                        loss += criterion(pred, label).item()
                        num = num + 1
                    avgloss = loss / (num - 1)
                    print('Loss/validation:' + str(avgloss))
                    # 保存loss值最小的网络参数
                    if avgloss < best_loss:
                        best_epoch = best_epoch + 1
                        best_loss = avgloss
                        best_loss_list.append(best_loss)
                        best_accuracy_list.append(1.0 - best_loss)
                        best_epochs.append(best_epoch)
                        save_model = param.model_pth
                        torch.save(net.state_dict(), save_model)

        # 绘制loss,accuracy的折线图
        plot1 = plt.plot(epoch_list, train_loss_list, label='Loss')
        plot2 = plt.plot(epoch_list, train_accuracy_list, label='Accuracy')

        plt.title('Loss & Accuracy of Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy')

        plt.legend(['Loss', 'Accuracy'])
        fig_name = str(args.K) + '-folds/%d_fold_' % (
                ki + 1) + args.net_type + '_%d_epoch_%.3f_loss_' % (
                       args.epochs, best_loss) + opt_type + '_optim.jpg'
        plt.savefig(fig_name)
        plt.close()

    plot3 = plt.plot(best_epochs, best_loss_list, label='Best Loss')
    plot4 = plt.plot(best_epochs, best_accuracy_list, label='Best Accuracy')
    plt.title('Loss & Accuracy of Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend(['Loss', 'Accuracy'])
    plt.savefig(args.net_type + '_%d_epoch_' % args.epochs + opt_type + '_optim.jpg')
    plt.close()


def main(net, device, args):
    train_net(net, device, args)


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    parser.add_argument('--net_type', type=str, default=param.model + param.loss, help='Choose Nets')
    parser.add_argument('--data_path', type=str, default="./", help='training data path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--val_epoch', type=int, default=5, help='Number of epochs to val.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--K', type=int, default=5, help='Number of cross validation')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate initial')
    parser.add_argument('--lrf', type=float, default=0.01, help='Scheduler Learning rate modify')
    parser.add_argument('--lf', type=float, default=0.001, help='Learning rate final')

    args = parser.parse_args()

    net = param.create_model
    # 释放无关内存
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # initial
    init_weights(net, init_type='kaiming')
    # 将网络拷贝到deivce中
    net.to(device=device)

    main(net, device, args)
