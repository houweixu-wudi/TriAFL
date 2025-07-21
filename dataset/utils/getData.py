import numpy as np
from torchvision import transforms as transforms
import utils.tools as tools
import torch
import random

class GetDataSet(object):
    def __init__(self, dataSetName):
        """
            dataSetName : 'PD'
            数据说明 : test的数据和标签都是tensor，其他都是列表，列表元素为tensor
        """
        self.name = dataSetName
        self.private_data = None # 训练集
        self.private_label = None # 标签

        self.public_data = None     # 公共数据
        self.public_label = None    # 公共数据标签

        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

        # self.test_data = None   # 测试数据集
        # self.test_label = None  # 测试的标签
        
    
        # 如何数据集是mnist
        if self.name == 'mnist':
            pass
        elif self.name == 'cifar10':
            pass
        elif self.name == "PD":
            self.loadPD()
        elif self.name == "TMMPD":
            self.loadTMMPD()
        elif self.name == "TMMPD_fivefolder":
            self.loadTMMPD_fivefolder()
        else:
            pass
  

    #加载PD数据集
    def loadPD(self):
        
        # 加载数据集
        pddatapath = ["../../PD/PDFace_unified"]     #91
        nopddatapaths = ["../../PD/TFED_Young", "../../PD/Oulu_cut", "../../PD/RaFD-cut"]
        # 加载私有的Pd数据集
        self.private_data, self.private_label = self.loaddata_by_paths(pddatapath, 1)
       
        # 加载公共数据集
        self.public_data, self.public_label = self.loaddata_by_paths(nopddatapaths, 0)

        # test_private_num = 25
        # self.private_data,self.test_private_data = private_data[:-test_private_num],private_data[-test_private_num:]
        # self.private_label,self.test_private_label = private_label[:-test_private_num],private_label[-test_private_num:]
        

        # test_public_num = 20
        # self.public_data,test_public_data = public_data[:-test_public_num],public_data[-test_public_num:]
        # self.public_label,test_public_label = public_label[:-test_public_num],public_label[-test_public_num:]
        
        # self.test_data,self.test_label =  torch.cat(test_private_data+test_public_data,dim=0),torch.cat(test_private_label+test_public_label,dim = 0)

        # self.test_data_size = len(self.test_data)

    def loadTMMPD(self):
        pd_test_num = 19
        pd_data, pd_label = self.loaddata_by_paths(["../../PD/PDFace_unified"], 1)

        pd_data_train , pd_data_test = pd_data[:-pd_test_num], pd_data[-pd_test_num:]
        pd_label_train, pd_label_test = pd_label[:-pd_test_num], pd_label[-pd_test_num:]

        public_data_test, public_label_test = self.loaddata_by_paths(["../../PD/TFED_Old"], 0)
        self.test_data = public_data_test + pd_data_test
        self.test_label = public_label_test + pd_label_test
        
        public_data_train, public_label_train = self.loaddata_by_paths(["../../PD/TFED_Young", "../../PD/Oulu_cut", "../../PD/RaFD-cut", "../../PD/ckplus"], 0)
        self.train_data = public_data_train + pd_data_train
        self.train_label = public_label_train + pd_label_train

    def loadTMMPD_fivefolder(self):
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []

        pd_data, pd_label = self.loaddata_by_paths(["../../PD/PDFace_unified"], 1)

        fivefloder = self.split_into_five_parts(list(zip(pd_data, pd_label)))
        public_data_test, public_label_test = self.loaddata_by_paths(["../../PD/TFED_Old"], 0)
        public_data_train, public_label_train = self.loaddata_by_paths(["../../PD/TFED_Young", "../../PD/Oulu_cut", "../../PD/RaFD-cut", "../../PD/ckplus"], 0)

        for i in range(len(fivefloder)):
            pd_data_test , pd_label_test = fivefloder[i]["data"], fivefloder[i]["label"]
            pd_data_train = []
            pd_label_train = []
            # 把其他折合并作为训练集
            for j in range(len(fivefloder)):
                if(j != i):
                    pd_data_train += fivefloder[j]["data"]
                    pd_label_train += fivefloder[j]["label"]
            
            self.test_data.append(public_data_test + pd_data_test)
            self.test_label.append(public_label_test + pd_label_test)
            
            self.train_data.append(public_data_train + pd_data_train)
            self.train_label.append(public_label_train + pd_label_train)

    # 加载数据
    def loaddata_by_paths(self, datapaths, isPD):
        """
        datapath : 数据路径,列表形式
        isPD : 0代表是非Pd数据，1代表是pd数据
        """
        data = []
        label = []
        for datapath in datapaths:
            data_temp, label_temp = tools.load_Personimages_from_folder(datapath, isPD)
            data, label = data_temp+data,label_temp+label
        # 打乱数据
        combined = list(zip(data, label))
        random.shuffle(combined)
        list1_shuffled, list2_shuffled = zip(*combined)
        data, label = list(list1_shuffled),list(list2_shuffled)
        return data, label
    
    def split_into_five_parts(self, lst):
        results = []
        k = 5
        # 计算每部分的大小
        n = len(lst)
        part_size = n // 5
        remainder = n % 5
        
        parts = []
        start = 0
        
        for i in range(k):
            # 基于余数调整最后几部分的大小
            end = start + part_size + (1 if i < remainder else 0)
            parts.append(lst[start:end])
            start = end

        # 对列表拆分，存入字典
        for i in range(k):
            temp_data, temp_label = zip(*parts[i])
            temp_data, temp_label = list(temp_data), list(temp_label)
            results.append({"data":temp_data, "label":temp_label})
        return results

if __name__=="__main__":
    'test data set'
    mydata = GetDataSet('PD') # test NON-IID
    print('the shape of the private data set is {}'.format(len(mydata.private_data)))
    print('the shape of the public data set is {}'.format(len(mydata.public_data)))
    # print('the shape of the test data set is {}'.format(len(mydata.test_data)))

