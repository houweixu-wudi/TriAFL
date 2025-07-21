import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image, ImageEnhance


# 读取RGB图片，以四维tensor形式返回   
def load_images_from_folder(folder,transform=None):
    # folder : 图片文件夹路径
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')
            img = transforms.ToTensor()(img)

            if transform is not None:
                img = transform(img)

            images.append(img)
    return torch.stack(images)


# 把同一个人的不同表情编为一组，进行读取
def load_Personimages_from_folder(folder,IsPD,transform=None):
    """
        params:
            folder : 图片文件夹路径
            transform : transform对象，对图片的处理
            返回的数据类型 : 为一个tensor数组，每一个tensor元素存储了一个人的所有表情
    """
    img = None
    images = []
    images_label = []
    temp = []
    # 读取文件夹中的文件名称,并排序
    filelist = sorted(os.listdir(folder))
    # 把第一个文件作为初始标记
    tagnamelist = filelist[0].replace('-', '_').split('_')
    for filename in filelist:
        # 读取图片
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            if transform is not None:
                img = transform(img)
            img = transforms.ToTensor()(img)

        # 把相同人的表情放在一起
        filenamelist = filename.replace('-', '_').split('_')
        if filenamelist[0] == tagnamelist[0]:
            temp.append(img)
        else:
            if IsPD:
                images_label.append(torch.ones(len(temp),dtype=torch.long))
            else:
                images_label.append(torch.zeros(len(temp),dtype=torch.long))
            images.append(torch.stack(temp))
            temp=[]
            temp.append(img)
            tagnamelist = filenamelist
    # 把最后一个人加进来
    if IsPD:
        images_label.append(torch.ones(len(temp),dtype=torch.long))
    else:
        images_label.append(torch.zeros(len(temp),dtype=torch.long))
    images.append(torch.stack(temp))
    return images,images_label


# def split_byclient(data, lable, split_sizes):
    
#     split_sizes = split_number_by_ratio(data.shape[0],split_sizes)  #确保相加为数据总数
#     split_data = torch.split(data, split_sizes)
#     split_lable = torch.split(lable, split_sizes)
#     return split_data,split_lable

def split_list_by_ratio(my_list,ratios):
    """
        按照比例划分列表，生成一个二维列表
    """
    # 计算划分点
    split_points = [int(len(my_list) * ratio) for ratio in ratios]
    partitions = [0] + [sum(split_points[:i+1]) for i in range(len(split_points))]

    # 使用划分点对列表进行划分
    sublists = [my_list[partitions[i]:partitions[i+1]] for i in range(len(partitions)-1)]
    
    return sublists
    
def split_byclient(my_list,ratios):
    """
    """
    sublists = split_list_by_ratio(my_list,ratios)
    sublists = [torch.cat(sublists[i],dim=0) for i in range(len(sublists))]
    
    return sublists


# 确保按比例ratios划分后，相加仍未x
def split_number_by_ratio(x, ratios):
    """
        x为元素之和
        ratios为划分比例
    """
    # 计算各个部分的大小
    parts = [int(x * ratio) for ratio in ratios]
    
    # 确保划分后的部分之和等于 x
    diff = x - sum(parts)
    parts[-1] += diff  # 将剩余的部分加到最后一个部分上
    
    return parts


# 拼接pd和非pd数据后，并返回相应的标签，1是pd，0是非pd
def catData(PdData,noPdData):
    #参数说明
    images = torch.cat((PdData, noPdData), dim=0)
    labels = torch.cat((torch.ones(PdData.shape[0],dtype=torch.long), torch.zeros(noPdData.shape[0],dtype=torch.long)), dim=0)
    return images,labels

# 将一张三通道的tensor图片进行增强
def adjust_img(img,way,factor):
    """
    img:一张三通道的tensor图片
    way:调整方式,-1不变，0饱和度，1亮度，2对比度
    factor:调整强度
    """
    img = transforms.functional.to_pil_image(img)
    if way == -1:
        pass
    elif way == 0:
        # 改变饱和度
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(factor)  # 增加饱和度两倍
    elif way == 1:
        # 改变亮度
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)  # 增加亮度50%
    elif way == 2:
        # 改变对比度
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)  # 减少对比度50%
    else:
        raise KeyError

    return transforms.functional.to_tensor(img)

def adjust_imgs(imgs,way,factor):
    """
    imgs:一个4维的batchsize的三通道的tensor图片
    way:调整方式，0饱和度，1亮度，3对比度
    factor:调整强度
    """
    transformed_images = torch.stack([adjust_img(img,way,factor) for img in imgs])
    return transformed_images





if __name__=="__main__":

    x,y = load_Personimages_from_folder("/home/houweixu/demo/DL/PD/TFED_cut",0)
    for i in y:
        if i.shape[0] != 8:
            print(i.shape[0])
    # print("\n\n\n")
    # y = load_Personimages_from_folder(folder="/home/houweixu/demo/DL/PD/PDFace_unified")
    print(len(x))
    # print(len(y))

