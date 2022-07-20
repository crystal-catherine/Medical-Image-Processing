import torch
import argparse
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torchvision import transforms
from data_loader import *
from nets import U_Net, Upp, AttU_Net, R2U_Net, R2AttU_Net, Upp, TransUnet, U_Transformer, MedT
from nets import init_weights
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim.lr_scheduler as lr_scheduler
from score import DiceLoss


def train_net(net, device, args):
    best_loss_list = []
    best_accuracy_list = []
    best_epochs = []

    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    best_epoch = 0

    # 定义优化算法
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-5)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.lf)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 定义Loss算法
    # no sigmoid
    # criterion = nn.BCEWithLogitsLoss()
    # last layer is sigmoid
    # criterion = nn.BCELoss()

    criterion = DiceLoss()

    # criterion1 = DiceLoss()
    # criterion2 = nn.BCEWithLogitsLoss()

    # ToTensor：img(H, W, C)->[0,1]tensor(C, H, W)
    # Normalize：[-1,1]
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5], [0.5])])

    for ki in range(args.K):
        # 加载训练集
        train_dataset = Data_Loader(args.data_path, ki=ki, K=args.K, typ='train', rand=True, img_size=args.img_size,
                                   imgcha=args.img_cha, transform=data_transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   )
        val_dataset = Data_Loader(args.data_path, ki=ki, K=args.K, typ='val', rand=True, img_size=args.img_size,
                                  imgcha=args.img_cha, transform=data_transform)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=args.batch_size,
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
            for image, label in train_loader:
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

                # loss1 = criterion1(pred, label)
                # loss2 = criterion2(pred, label)
                # loss = 0.5 * loss1 + 0.5 * loss2
                # loss_list.append(loss)
                # loss_sum += loss.item()

                # 更新参数
                loss.backward()
                optimizer.step()
                count = count + 1
            print(scheduler.get_last_lr())

            # if (epoch + 1) % 2 == 0:
            #     scheduler.step()
            scheduler.step()
            epoch_loss = loss_sum / count
            print('Loss/train', epoch_loss)
            epoch_accuracy = 1.0 - epoch_loss
            train_loss_list.append(epoch_loss)
            train_accuracy_list.append(epoch_accuracy)

            if (epoch + 1) % 5 == 0:
                net.eval()
                with torch.no_grad():
                    num = 1
                    loss = 0.
                    for image, label in val_loader:
                        image = image.to(device=device, dtype=torch.float32)
                        label = label.to(device=device, dtype=torch.float32)
                        pred = net(image)

                        # loss1 = criterion1(pred, label)
                        # loss2 = criterion2(pred, label)
                        # loss += (0.5 * loss1 + 0.5 * loss2).item()

                        loss += criterion(pred, label).item()
                        num = num + 1
                    avgloss = loss / (num - 1)
                    print('Loss/validation', avgloss)
                    # 保存loss值最小的网络参数
                    if avgloss < best_loss:
                        best_epoch = best_epoch + 1
                        best_loss = avgloss
                        best_loss_list.append(best_loss)
                        best_accuracy_list.append(1.0 - best_loss)
                        best_epochs.append(best_epoch)
                        save_model = './model/' + args.net_type + '_model_' + args.opt_type + '_epoch' + str(
                            args.epochs) + '.pth'
                        torch.save(net.state_dict(), save_model)

        # 绘制loss,accuracy的折线图
        plot1 = plt.plot(epoch_list, train_loss_list, label='Loss')
        plot2 = plt.plot(epoch_list, train_accuracy_list, label='Accuracy')

        plt.title('Loss & Accuracy of Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy')

        plt.ylim(0.0, 1.)
        plt.legend(['Loss', 'Accuracy'])
        fig_name = str(args.K) + '-folds/%d_fold_' % (
                ki + 1) + args.net_type + '_%d_epoch_%.3f_loss_' % (
                       args.epochs, best_loss) + args.opt_type + '_optim.jpg'
        plt.savefig(fig_name)
        plt.close()

    plot3 = plt.plot(best_epochs, best_loss_list, label='Best Loss')
    plot4 = plt.plot(best_epochs, best_accuracy_list, label='Best Accuracy')
    plt.title('Loss & Accuracy of Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.ylim(0.0, 1.)
    plt.legend(['Loss', 'Accuracy'])
    plt.savefig(args.net_type + '_%d_epoch_' % args.epochs + args.opt_type + '_optim.jpg')
    plt.close()


def main(net, device, args):
    train_net(net, device, args)


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    parser.add_argument('--net_type', type=str, default='TransUNet12_green_Dice', help='Choose Nets')
    parser.add_argument('--opt_type', type=str, default='Adam_cos', help='Optimizer_scheduler')
    parser.add_argument('--data_path', type=str, default="./Dataset/process_train/", help='training data path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--T_max', type=int, default=10, help='Half of scheduler T.')
    parser.add_argument('--K', type=int, default=10, help='Number of cross validation')
    parser.add_argument('--img_size', type=int, default=512, help='Resize image')
    parser.add_argument('--img_cha', type=int, default=1, help='Image channel')
    parser.add_argument('--img_class', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate initial')
    parser.add_argument('--lrf', type=float, default=0.01, help='Scheduler Learning rate modify')
    parser.add_argument('--lf', type=float, default=1e-4, help='Learning rate final')

    args = parser.parse_args()

    # 加载网络，图片单通道1，分类为1。
    # net = U_Net(img_ch=args.img_cha, output_ch=args.img_class)
    # net = Upp(n_channels=args.img_cha, n_classes=args.img_class)
    # net = AttU_Net(img_ch=args.img_cha, output_ch=args.img_class)
    # net = R2U_Net(img_ch=args.img_cha, output_ch=args.img_class)
    # net = R2AttU_Net(img_ch=args.img_cha, output_ch=args.img_class)
    net = TransUnet(in_channels=args.img_cha, img_dim=args.img_size, vit_blocks=12, vit_dim_linear_mhsa_block=args.img_size,
                    classes=args.img_class)
    # net = MedT(img_size=args.img_size, imgchan=args.img_cha, num_classes=args.img_class)
    # 释放无关内存
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # initial
    init_weights(net, init_type='kaiming')
    # 将网络拷贝到deivce中
    net.to(device=device)

    # 指定训练集地址，开始训练
    main(net, device, args)
