import time
from flcore.clients.clientrt import clientRT
from flcore.servers.serverbase import Server
from threading import Thread
import copy
import torch
import numpy as np
import os


class FedRT(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.fedce_cos_sim = {}
        self.fedce_coef = [1.00/self.num_clients for _ in range(self.num_clients)]
        self.fedce_minus_vals=[]
        self.fedce_client_speed = []
        self.fedce_mode = "plus"
        self.weight_rates =  [x_rate / 21.00 for x_rate in [10,8,3]]    #贡献估计的各个贡献的权重分配

        self.weight_way = args.weight_way

        # 替换batchnorm层
        # self.replace_batchnorm(self.global_model)
        # self.replace_batchnorm(args.model)
        # print(args.model)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientRT)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.clientdatanum = []
        self.getclientdatanum()
        
        
    def setWeigth_rates(self,weight):
        """
        weight:权重比例
        """
        self.add_to_json(os.path.join(self.records_dir, self.tb_nowtime, 'config.json'), {"weight_rates":weight})
        self.weight_rates = weight/np.sum(weight)

    def train(self):
        for i in range(self.global_rounds+1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if(self.weight_way==2):
                ##
                # send params to clients and get params from clients
                self.fedce_minus_vals = []
                self.fedce_client_speed = []
                for idx,client in enumerate(self.clients):
                    client.current_round = i
                    # round 0, initialize uniform fedce_coef
                    client.fedce_coef = self.fedce_coef[idx]
                ##

            # for key, value in self.global_model.state_dict().items():
            #     if 'bn1' in key:
            #         print(key,":"+'-'*5,value)
            #         break

            # 冻结参数
            # if(i == 10000):
            #     self.fixBatchnorm()
            if(i == self.global_rounds/2):
                print("Frizon parameter"+"*"*10)
                self.fixBatchnorm()

            self.send_models()
            self.send_batchsize()
            # self.lr_scheduler()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)


            if(self.weight_way==2):
                ##
                self.getuploaded_weights()
            elif(self.weight_way == 0):
                raise ValueError("不能直接相加，客户端上传的是整个模型")
                for idx in range(len(self.uploaded_weights)):
                    self.uploaded_weights[idx] = 1


            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # self.save_results()
        # self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientRT)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def getclientdatanum(self):
        for client in self.clients:
            self.clientdatanum.append(len(client.train_dataset))

    def send_batchsize(self):
        total = 0.0
        tenp_clients = []

        for client in self.selected_clients:
            total = total + self.clientdatanum[client.id]

        for client in self.selected_clients:
            temp_batch = int(self.clientdatanum[client.id] / total * self.batch_size)
            # print(temp_batch)
            if temp_batch == 0 :
                client.batch_size = int(total / self.batch_size) #本地dataloader相当于为空了，也即是抛弃这个客户端
            else:
                client.batch_size = temp_batch

            # if temp_batch != 0 :
            #     client.batch_size = temp_batch
            #     tenp_clients.append(client)

        # self.selected_clients = tenp_clients
        # self.current_num_join_clients = len(self.selected_clients)

    def getuploaded_weights(self):

        for client_idx in self.uploaded_ids:
            self.fedce_minus_vals.append(self.clients[client_idx].minus_val)
            self.fedce_client_speed.append(self.clients[client_idx].speed_val)

        # generate consensus gradient with current FedCE coefficients
        consensus_grad = []
        global_weights = self.global_model.state_dict()
        for idx, name in enumerate(global_weights):
            temp = torch.zeros_like(global_weights[name],dtype=float)
            for idx,client_model in enumerate(self.uploaded_models):
                temp += self.fedce_coef[self.uploaded_ids[idx]] * client_model.state_dict()[name]
            consensus_grad.append(temp.data.view(-1))

        # flatten for cosine similarity computation
        consensus_grads_vec = torch.cat(consensus_grad).to("cpu")

        # generate minus gradients and compute cosine similarity
        self.fedce_cos_sim[self.current_round] = {}

        for idx,client_model in enumerate(self.uploaded_models):
            site_grad = []
            for name in self.global_model.state_dict():
                site_grad.append(client_model.state_dict()[name].data.view(-1))
            site_grads_vec = torch.cat(site_grad).to("cpu")
            # minus gradient
            minus_grads_vec = consensus_grads_vec - self.fedce_coef[self.uploaded_ids[idx]] * site_grads_vec
            # compute cosine similarity
            fedce_cos_sim_site = (
                torch.cosine_similarity(site_grads_vec, minus_grads_vec, dim=0).detach().cpu().numpy().item()
            )
            # append to record dict
            self.fedce_cos_sim[self.current_round][self.uploaded_ids[idx]] = fedce_cos_sim_site

        # compute cos_weights and minus_vals based on the record for each site
        fedce_cos_weights = []
        for site in self.uploaded_ids:
            # cosine similarity
            cos_accu_avg = np.mean([self.fedce_cos_sim[i][site] for i in range(len(self.fedce_cos_sim))])
            fedce_cos_weights.append(1.0 - cos_accu_avg)
        
        # normalize
        fedce_cos_weights /= np.sum(fedce_cos_weights)
        fedce_cos_weights = np.clip(fedce_cos_weights, a_min=1e-3, a_max=None)
        self.fedce_minus_vals /= np.sum(self.fedce_minus_vals)
        self.fedce_minus_vals = np.clip(self.fedce_minus_vals, a_min=1e-3, a_max=None)
        self.fedce_client_speed /= np.sum(self.fedce_client_speed)
        self.fedce_client_speed = np.clip(self.fedce_client_speed, a_min=1e-3, a_max=None)
        
        # two aggregation strategies
        # if self.fedce_mode == "times":
        #     new_fedce_coef = [c_w * mv_w for c_w, mv_w in zip(fedce_cos_weights, self.fedce_minus_vals)]
        # elif self.fedce_mode == "plus":
        #     new_fedce_coef = [c_w + mv_w for c_w, mv_w in zip(fedce_cos_weights, self.fedce_minus_vals)]
        # else:
        #     raise NotImplementedError
        
        if self.fedce_mode == "times":
            new_fedce_coef = [c_w * mv_w*cs_w for c_w, mv_w,cs_w in zip(fedce_cos_weights, self.fedce_minus_vals,self.fedce_client_speed)]
        elif self.fedce_mode == "plus":
            new_fedce_coef = [c_w*self.weight_rates[0] + mv_w*self.weight_rates[1]+cs_w*self.weight_rates[2] for c_w, mv_w,cs_w in zip(fedce_cos_weights, self.fedce_minus_vals,self.fedce_client_speed)]
        else:
            raise NotImplementedError

        # normalize again

        new_fedce_coef /= np.sum(new_fedce_coef)
        new_fedce_coef = np.clip(new_fedce_coef, a_min=1e-3, a_max=None)
        
        # update fedce_coef
        # fedce_coef = {}
        idx = 0
        for site in self.uploaded_ids:
            self.fedce_coef[site] = new_fedce_coef[idx]
            idx += 1
        
        self.uploaded_weights = new_fedce_coef
    
    # 客户端只上传了梯度，这个模型有batchnorm层时，下面这种做法是有问题的
    # def aggregate_parameters(self):
    #     assert (len(self.uploaded_models) > 0)
  
    #     for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
    #         for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
    #             server_param.data += client_param.data.clone() * w


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        # for param in self.global_model.parameters():
        #     param.data.zero_()

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for key, param in self.global_model.state_dict().items():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        # for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
        #     server_param.data += client_param.data.clone() * w

        for key, server_param in self.global_model.state_dict().items():
            # 获取 server_param.data 的数据类型
            data_type = server_param.data.dtype
            # 进行运算并确保所有操作都使用相同的数据类型
            if data_type == torch.float32:  # 如果 server_param.data 是浮点数
                server_param.data += client_model.state_dict()[key].clone().float() * w
            elif data_type == torch.int64:  # 如果 server_param.data 是整数
                server_param.data += (client_model.state_dict()[key].clone().long() * w).long()
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
    
    def fixBatchnorm(self):
        # for key, value in self.global_model.state_dict().items():
        #     if 'bn' in key:
        #         value.requires_grad = False

        for client in self.clients:
            # for layer in client.model.modules():
            #     if isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    # print(layer)
                    # layer.track_running_stats = False
                    # for param in layer.parameters():
                    #     param.requires_grad = False

            # 冻结batchnorm时  
            client.frozenBatchNorm = True
            # 调小学习率，不然可能会爆炸  
            # for opt_param in client.optimizer.param_groups:
            #     opt_param["lr"] = 0.005

    
    # 改变学习率
    def lr_scheduler(self):
        lr_decay = 0.998
        for client in self.clients:
            for opt_param in client.optimizer.param_groups:
                    opt_param["lr"] *= lr_decay