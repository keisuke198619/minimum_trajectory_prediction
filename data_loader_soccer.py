import torch
from torch.utils.data import Dataset
from importlib import import_module
import cloudpickle

class Dataset(Dataset):

    def __init__(self, args, len_seqs, train=0, normalize_data=True):
        #if normalize_data:
        #    for index in range(len(activeRole)):
        #        data_x[index] = dataset.normalize(data[index],datasets)

        self.train = train
        self.len_seqs = len_seqs
        self.game_files = args.game_files
        self.game_files_val = args.game_files_val
        self.game_files_te = args.game_files_te
        self.batchsize = args.batchsize
        self.model = args.model
        self.n_GorS = args.n_GorS

        #self.normalize_data = normalize_data
        #self.datasets = args.dataset
 
    def __getitem__(self, index):
        if self.train == 1:
            batch_no = index // self.batchsize
            batch_index = index % self.batchsize
            with open(self.game_files+'_tr'+str(batch_no)+'.pkl', 'rb') as f:
                data,_,_ = cloudpickle.load(f) # ,allow_pickle=True
        else:
            J = 4 if self.n_GorS == 1 else 8
            batch_no = index // int(self.len_seqs/J)
            if batch_no < J:
                batch_index = index % int(self.len_seqs/J)
            else:
                batch_no = J-1
                batch_index = index % int(self.len_seqs/J) + int(self.len_seqs/J)
            filename = self.game_files+'_val' if self.train == 0 else self.game_files +'_te'
            with open(filename+'_'+str(batch_no)+'.pkl', 'rb') as f:
                data = cloudpickle.load(f) 

        if len(data) == 1: # tentative
            data = data[0]

        self.data = torch.Tensor(data)
        self.data = self.data.permute(1, 0, 2, 3) # batch,agents,length,dim


        return self.data[batch_index] # data[index] 

    def __len__(self):
        return self.len_seqs
        