"""
Set up COVID speech datasets.
"""

import os,sys
sys.path.append("./Local/")
import torch
import torchaudio
import pickle
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from Utils import util
import ML_func

# %% DiCOVA2 
class DiCOVA_2_Dataset(torch.utils.data.Dataset):
    def __init__(self,args):
		
        self.file_list = args['file_list']
        self.label_file = args['label_file']
        self.dataset_args = args['dataset_args']
        self.augmentaion_args = args.get('augmentation_args',None)
        self.shuffle = args.get('shuffle',False)
        self.device = args['device']

        self.mode = self.dataset_args['mode']
        if self.mode == 'segment':        
            self.segment_length = self.dataset_args['segment_length']
            self.segment_hop = self.dataset_args['segment_hop']
        self.oversampling = self.dataset_args.get('oversampling',False)
        if self.oversampling:
            self.oversampling_factor = self.dataset_args.get('oversampling_factor',-1)
        self.apply_mean_norm = self.dataset_args.get('apply_mean_norm',False)
        self.apply_var_norm = self.dataset_args.get('apply_var_norm',False)

        self.augment=False
        if self.augmentaion_args:
            if self.augmentaion_args['mode']=='masking':
                self.augment=True
                self.freq_mask=torchaudio.transforms.FrequencyMasking(self.augmentaion_args['freq_mask_param'])
                self.time_mask=torchaudio.transforms.TimeMasking(self.augmentaion_args['time_mask_param'])

        self.generate_examples()

    def generate_examples(self):

        file_list = open(self.file_list).readlines()
        file_list = [line.strip().split() for line in file_list]
        file_paths = {}

        for line in file_list:
            file_paths[line[0]]=line[1]

        temp = open(self.label_file).readlines()
        temp = [line.strip().split() for line in temp]
        labels={}
        categories = ['n','p']
        for fil,label in temp:
            labels[fil]=categories.index(label)
        del temp

        if self.oversampling and self.oversampling_factor==-1:
            # If oversampling_factor is not specified, compute it automatically
            l = np.array(list(labels.values()))
            l = int(len(np.where(l==0)[0])/len(np.where(l==1)[0]))-1
            self.oversampling_factor=l

        egs = []
        for fil in list(labels.keys()):
            path = file_paths[fil]
            F = pickle.load(open(path,'rb'))

            # Apply utterance level normalization
            if self.apply_mean_norm: F = F - torch.mean(F,dim=0)
            if self.apply_var_norm: F = F / torch.std(F,dim=0)

            label = labels.get(fil,None)
            egs.append( (F.to(self.device),torch.FloatTensor([label]).to(self.device)))
            if label==1 and self.oversampling:
                # Oversample positive class samples only
                # Diversity in examples is created by random cropping 	
                for i in range(self.oversampling_factor):	
                    nF=int((0.8+0.2*np.random.rand())*F.shape[0])
                    start=max(0,int(np.random.randint(0,F.shape[0]-nF,1)))
                    egs.append((F[start:start+nF,:].to(self.device),torch.FloatTensor([label]).to(self.device)))

        # File mode and Segment mode
        if self.mode=='file':
            egs=egs
        elif self.mode=='segment':
            fegs=[]
            for F,L in egs:
                start_pt=0;end_pt=min(F.shape[0],self.segment_length)
                while end_pt<=F.shape[0]:
                    fegs.append((F[start_pt:end_pt,:],L))
                    start_pt+=self.segment_hop;end_pt=start_pt+self.segment_length
            egs=fegs
        else:
            raise ValueError("Unknown mode for examples")

        # SpecAugment style augmentation
        # Frequency mask, time mask, and frequency-time mask applied separately 
        if self.augment:
            e1=[]
            for eg in egs:
                F,l = eg
                F = self.freq_mask(F)
                eg = (F,l)
                e1.append(eg)
            e2=[]
            for eg in egs:
                F,l = eg
                F = self.time_mask(F)
                eg = (F,l)
                e2.append(eg)
            e3=[]
            for eg in egs:
                F,l = eg
                F = self.freq_mask(F)
                F = self.time_mask(F)
                eg = (F,l)
                e3.append(eg)
            egs.extend(e1)
            egs.extend(e2)
            egs.extend(e3)
        if self.shuffle: random.shuffle(egs)
        self.egs=egs

    def __len__(self):
        return len(self.egs)
    
    def __getitem__(self, idx):
        feat, label = self.egs[idx]
        return feat, label

    def collate(self,batch):
        # A batch of examples is a list
        inputs = [t[0] for t in batch]
        targets = [t[1] for t in batch]
        return (inputs, targets)



class _covid_dataset(Dataset):
    """
    This class can be used to prepare datasets for all three COVID datasets and two types of
    features.
    """
    def __init__(self,\
                 dataset:str,
                 feat:str,
                 split:str,
                 metadata_path:str,
                 scale:str='standard',
                 filters:tuple=None):
        
        # sanity check
        assert split in ['train','valid','test']
        assert feat in ['logmelspec','mtr','mtr_v2','mtr_v3']
        assert dataset in ['CSS','DiCOVA2','Cambridge'], 'Unknown dataset!'

        # load desired features and labels
        data, label = util.load_feat(metadata_path,feat,split)
        # sanity check on number of features and labels
        assert data.shape[0] == label.shape[0], 'data size does not match label size!'
        
        # some pre-processing
        if 'mtr' in feat:
            # adjust axes for downstream model
            data = np.moveaxis(data,[1,2,3],[2,3,1])
            data = data[:,np.newaxis,...]
            data = 20*np.log10(data) # power to db
        elif feat == 'logmelspec':
            data = np.moveaxis(data,[1,2],[2,1])

        # scale data (file-level)
        if 'mtr' in feat: axis = (2,3,4)
        elif feat == 'logmelspec': axis = (1,2)
        if scale == 'minmax':
            data = ML_func.minmax_scaler(data,axis)
        elif scale == 'standard':
            data = ML_func.standard_scaler(data,axis) 

        # spectral-temporal saliency masks
        if filters is not None:
            data = ML_func.multi_filter(x=data,hei=23,wid=8,filters=filters)
        
        # value check
        assert np.nan not in data, "NaN in data"

        self.data = data
        self.label = label
    
    def __getitem__(self,idx):
        _data = torch.FloatTensor(self.data[idx])
        _label = torch.FloatTensor(self.label[idx])
        sample = {"data": _data, "label": _label}
        return sample
    
    def __len__(self):
        return self.data.shape[0]

#%%
if __name__ == '__main__':

    # data, label = util.load_feat('/mnt/d/projects/COVID-datasets/CSS/label/metadata_og.csv',\
    #                              'mtr',
    #                              'train')
    # print(label.shape)

    import matplotlib.pyplot as plt
    toy_dataset = _covid_dataset(dataset='DiCOVA2',\
                                 feat='logmelspec',
                                 split='train',
                                 metadata_path='/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_og.csv',
                                 scale='standard',
                                 filters=None)
    print(toy_dataset.label.shape)