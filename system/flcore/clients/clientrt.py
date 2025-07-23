import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
import copy
from utils.data_utils import Dataset

class clientRT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 0代表直接相加，1代表数量加权，2代码置信度估计
        self.weight_way = args.weight_way
        self.local_step = args.local_step
        ##
        self.fedce_minus_val = []
        self.lastmodel_trainval = 0
        self.fedce_localtrain_speed = []
        self.model_last_round = copy.deepcopy(self.model)
        self.fedce_coef = 0
        self.current_round = 0
        self.minus_val = 0
        self.speed_val = 0
        ##
        self.train_dataset = self.load_train_dataset()
        self.frozenBatchNorm = False
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.001)  #替换优化器，采用衰退

    def train(self):
        
        
        if(self.weight_way==2):
            ##
            if self.current_round > 0:
                # get FedCE coefficient
                
                # get fedce_minus model
                fedce_minus_model = self.get_minus_model(
                    self.model,
                    self.model_last_round,
                    self.fedce_coef,
                )
                # validate minus model
                # minus_metric,_ = self.local_valid(
                #     fedce_minus_model,
                #     self.testloaderfull
                # )

                minus_metric,_ = self.local_valid(
                    fedce_minus_model,
                    self.trainloader
                )
                
                # add to the record
                self.fedce_minus_val.append(minus_metric)
            else:
                self.fedce_coef = 0.0
                self.fedce_minus_val.append(0.0)
            ##

        modelbefore = copy.deepcopy(self.model)

        trainloader = DataLoader(Dataset(data=self.train_dataset, dataset_name=self.dataset), self.batch_size, drop_last=True, shuffle=True)

        # trainloader = DataLoader(self.train_dataset, self.batch_size, drop_last=True, shuffle=True)
        # self.model.to(self.device)
        self.model.train()
        # 冻结Batchnorm参数
        if(self.frozenBatchNorm):
            for m in self.model.modules():
                if isinstance(m,nn.BatchNorm2d):
                    m.eval()
                    
        start_time = time.time()

        if(self.local_step == 0):
            self.local_step = int(len(self.train_dataset) / self.batch_size * self.local_epochs)

        total_step = 0

        # if self.id == 1:
        #     for key, value in self.model.state_dict().items():
        #         print(key,":"+'-'*5,value[0])
        #         break

        while(total_step < self.local_step):
            if(len(trainloader) == 0):
                break
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10) 
                
                self.optimizer.step()

                total_step = total_step + 1
                if(total_step == self.local_step):
                    break
            if(total_step == self.local_step):
                break

        # if self.id == 1:
        #     for key, value in self.model.state_dict().items():
        #         print(key,":"+'-'*5,value[0])
        #         break

        if(self.weight_way==2):
            ##
            # compute speed contribution
            _,currentmodel_trainval = self.local_valid(self.model,self.trainloader)
            if(currentmodel_trainval == 0):
                current_speedcoef = 0
            else:
                current_speedcoef = -(currentmodel_trainval - self.lastmodel_trainval)/currentmodel_trainval
            self.fedce_localtrain_speed.append(current_speedcoef)

            self.lastmodel_trainval = currentmodel_trainval
            self.speed_val = np.mean(self.fedce_localtrain_speed)
            
            # compute delta models, initial models has the primary key set
            local_weights = self.model.state_dict()
            # use the historical mean of minus_val for FedCE
            self.minus_val = 1.0 - np.mean([self.fedce_minus_val[i] for i in range(len(self.fedce_minus_val))])

            # update model_last_round
            self.model_last_round.load_state_dict(local_weights)
            ##

        # 相减，只上传梯度，注意模型要是有batchnorm层，这种相减方式是错误的
        # for toserver_param, client_param in zip(self.model.parameters(), modelbefore.parameters()):
        #         toserver_param.data -= client_param.data.clone()
        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def load_train_dataset(self):
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return train_data
  
    def get_minus_model(self, global_model, last_round_model, fedce_weight):
        minus_model = copy.deepcopy(global_model)
        for key in minus_model.state_dict().keys():
            temp = (global_model.state_dict()[key] - fedce_weight * last_round_model.state_dict()[key]) / (
                1 - fedce_weight
            )
            minus_model.state_dict()[key].data.copy_(temp)
        return minus_model
    
    def local_valid(
        self,
        model,
        valid_loader
    ):
        """Typical validation logic
        Load data pairs from train_loader: image / label
        Compute outputs with self.model
        Perform post transform (binarization, etc.)
        Compute evaluation metric with self.valid_metric
        Add score to tensorboard record with specified id
        """
        model.eval()
        with torch.no_grad():
            loss_list = []
            metric = 0
            loss_mean = 0
            for i, (x,y) in enumerate(valid_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                preds = self.model(x)
                loss = self.loss(preds,y)
                loss_list.append(float(loss))
                preds = torch.argmax(preds, dim=1)
                metric += (preds == y).float().mean()
                
            # compute mean dice over whole validation set
            if(len(valid_loader) != 0):
                metric /= len(valid_loader)
                loss_mean = np.mean(loss_list)

        return float(metric),loss_mean
    
    def set_parameters(self, model):
        # for new_param, old_param in zip(model.parameters(), self.model.parameters()):
        #     old_param.data = new_param.data.clone()

        self.model.load_state_dict(model.state_dict(), strict=True)#
        # 聚合时排除batchnorm层
        # for key in model.state_dict().keys():
        #     if 'bn' not in key:
        #         self.model.state_dict()[key].data.copy_(model.state_dict()[key])

