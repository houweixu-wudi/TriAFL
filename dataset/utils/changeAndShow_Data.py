import numpy as np
import os
import torch
from utils.dataset_utils import read_data, getidx_num, read_data
from numpy.testing import assert_array_almost_equal
from utils.tools import adjust_img
import numpy
import matplotlib.pyplot as plt


def changeData(dataset, idx, way = 0, is_train = True, noise_rate = 0.1, classnum = 10, img_way = -1, factor = 1.0):
    """
    must: dataset, idx, way, is_train, noise_rate

    label noisy: classnum

    img: img_way,factor
    """
    data_dict = {}
    train_data = read_data(dataset, idx, is_train)
    X_data = torch.Tensor(train_data['x']).type(torch.float32)
    Y_data = torch.Tensor(train_data['y']).type(torch.int64)
    # train_data = [(x, y) for x, y in zip(X_data, Y_data)]
    print(X_data.shape)
    if(way == 0):
        noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(y_train=np.asarray(Y_data), noise = noise_rate, random_state=0, nb_classes=classnum)
        print(actual_noise_rate)

        Y_data = torch.tensor(noisy_labels)
    elif(way == 1):
        
        X_data = addnoise_to_image(X_data, noise_rate, img_way, factor)


    data_dict['x'] = X_data.numpy()
    data_dict['y'] = Y_data.numpy()

    if(is_train):
        data_dir = os.path.join(dataset, 'train/'+str(idx) + '.npz')
    else:
        data_dir = os.path.join(dataset, 'test/'+str(idx) + '.npz')

    delete_file(data_dir)

    with open(data_dir, 'wb') as f:
        np.savez_compressed(f, data=data_dict)


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"文件 {file_path} 已成功删除")
    except OSError as e:
        print(f"删除文件时出错: {e.filename} - {e.strerror}")

def addnoise_to_image(X_data, noise_rate, img_way, factor):
    """
    X_data为一个三维图片tensor
    """
    X_temp = []
    # 随机种子问题注意
    flipper = np.random.RandomState(0)
    P = [1-noise_rate, noise_rate]
    for img in X_data:
        # multinomial(x, y, z)表示以概率数组P掷骰子，掷x次，返回每个面（也就是每种情况）出现的次数，z表示执行z轮
        flipped = flipper.multinomial(1, P, 1)[0]
        if(np.where(flipped == 1)[0]):
            X_temp.append(adjust_img(img, img_way, factor))
        else:
            X_temp.append(img)
    X_data = torch.stack(X_temp)
    return X_data

# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print(m)
    new_y = y.copy()

    # 随机种子问题注意
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=0, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print(P)

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=0, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print(P)

    return y_train, actual_noise

def showlabel_distribution(dataset_name = "Cifar10_dir10", isTrain = True):
    print(dataset_name)
    clients_num = getidx_num(dataset_name)
    labels = []
    for i in range(clients_num):
        data = read_data(dataset_name, i, isTrain)
        labels.append(numpy.array(data['y']))
    
    # 以标签作为横轴
    # min_value = min([np.min(arr) for arr in labels])
    # max_value = max([np.max(arr) for arr in labels])
    # plt.figure(figsize=(12, 8))
    # plt.hist(labels, stacked=True,
    #             bins=np.arange(min_value-0.5, max_value + 1.5, 1),
    #             label=["Client {}".format(i) for i in range(len(labels))],
    #             rwidth=0.5)
    # plt.xticks(np.arange(clients_num), np.arange(clients_num))
    # plt.xlabel("Label type")
    # plt.ylabel("Number of samples")
    # plt.legend(loc="upper right")
    # plt.title("Display Label Distribution on Different Clients")
    # plt.show()


    ## 以客户端作为横轴
    max_value = max([np.max(arr) for arr in labels])
    label_distribution = [[] for _ in range(int(max_value+1))]
    for c_id, idc in enumerate(labels):
        for idx in idc:
            label_distribution[idx].append(c_id)
    labels = label_distribution

    min_value = min([np.min(arr) for arr in labels])
    max_value = max([np.max(arr) for arr in labels])
    plt.figure(figsize=(12, 8))
    plt.hist(labels, stacked=True,
                bins=np.arange(min_value-0.5, max_value + 1.5, 1),
                label=["label {}".format(i) for i in range(len(labels))],
                rwidth=0.5)
    plt.xticks(np.arange(clients_num), np.arange(clients_num))
    plt.xlabel("Clients")
    plt.ylabel("Number of samples")
    plt.legend(loc="upper right")
    plt.title("Display Label Distribution on Different Clients")
    plt.show()


def showimages(tensor_images):
    """
    tensor_images: 存储了多张三维rgb图片的tensor变量
    """
    N = tensor_images.shape[0]
    # 计算行和列的布局
    cols = 5  # 每行展示3张图片
    rows = int(np.ceil(N / cols))  # 总行数

    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()  # 拉平为1维数组，方便索引

    # 展示每一张图片
    for i in range(N):
       # 使用 permute 将形状从 (C, H, W) 转换为 (H, W, C)
        img = tensor_images[i].permute(1, 2, 0).numpy()  # 转换为 NumPy 数组
        axes[i].imshow(img)  # 显示第 i 张图片
        axes[i].set_title(f'Image {i + 1}')
        axes[i].axis('off')  # 不显示坐标轴

    # 如果有多余的子图，则隐藏它们
    for j in range(N, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    
    changeData(dataset="changedata", idx = 3, way = 0, is_train = True, noise_rate = 0.1, classnum = 10)
    # showdata()

