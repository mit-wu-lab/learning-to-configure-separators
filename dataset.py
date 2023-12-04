from abc import ABC, abstractmethod
import json
import os
import torch
import torch_geometric
from torch_geometric.data import Dataset
import pickle
import glob
import platform
from torch_geometric.data import Data

import data as _data
import utils as _utils
from copy import deepcopy as dp

class bandit_dataset(torch_geometric.data.Dataset):
    def __init__(self, train_set):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.train_set = dp(train_set)
    
    def len(self):
        return len(self.train_set)
    
    def get(self, index):
        self.train_set[index][0].labels = self.train_set[index][1]
        return bandit_data(
            self.train_set[index][0].x_rows,
            self.train_set[index][0].x_cols,
            self.train_set[index][0].x_sepas, 
            self.train_set[index][0].edge_index_rowcols,
            self.train_set[index][0].edge_vals_rowcols,
            self.train_set[index][0].edge_index_sepa_cols, 
            self.train_set[index][0].edge_vals_sepa_cols,
            self.train_set[index][0].edge_index_sepa_rows, 
            self.train_set[index][0].edge_vals_sepa_rows, 
            self.train_set[index][0].edge_index_sepa_self, 
            self.train_set[index][0].edge_vals_sepa_self,
            self.train_set[index][0].labels
        )

class offline_dataset(torch_geometric.data.Dataset):
    def __init__(self, train_set, intputs_context):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.train_set = dp(train_set)
        self.inputs_context = dp(intputs_context)
    
    def len(self):
        return len(self.train_set)
    
    def get(self, index, device='cuda:0'):
        def sc(index):
            return self.inputs_context[self.train_set[index][0]]
        return bandit_data(
            sc(index).x_rows,
            sc(index).x_cols,
            torch.tensor(self.train_set[index][1]).to(torch.float32), 
            sc(index).edge_index_rowcols,
            sc(index).edge_vals_rowcols,
            sc(index).edge_index_sepa_cols, 
            sc(index).edge_vals_sepa_cols,
            sc(index).edge_index_sepa_rows, 
            sc(index).edge_vals_sepa_rows, 
            sc(index).edge_index_sepa_self, 
            sc(index).edge_vals_sepa_self,
            torch.tensor(self.train_set[index][2]).to(torch.float32), 
        )

N_SEPAS = 17

class bandit_data(Data):
    
    def __init__(
        self,
        x_rows,
        x_cols,
        x_sepas, 
        edge_index_rowcols,
        edge_vals_rowcols,
        edge_index_sepa_cols, 
        edge_vals_sepa_cols,
        edge_index_sepa_rows, 
        edge_vals_sepa_rows, 
        edge_index_sepa_self, 
        edge_vals_sepa_self,
        labels
    ):  
        super().__init__()
        self.x_rows = x_rows
        self.x_cols = x_cols
        self.x_sepas = x_sepas
        self.edge_index_rowcols = edge_index_rowcols
        self.edge_vals_rowcols = edge_vals_rowcols
        self.edge_index_sepa_cols = edge_index_sepa_cols
        self.edge_vals_sepa_cols = edge_vals_sepa_cols
        self.edge_index_sepa_rows = edge_index_sepa_rows
        self.edge_vals_sepa_rows = edge_vals_sepa_rows
        self.edge_index_sepa_self = edge_index_sepa_self
        self.edge_vals_sepa_self = edge_vals_sepa_self
        self.labels = labels
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'x_cuts':
            inc = 0
        elif key == 'x_rows':
            inc = 0
        elif key == 'x_cols':
            inc = 0
        elif key == 'lpvals':
            inc = 0
        elif key == 'sepa_settings':
            inc = 0
        elif key == 'masks': 
            inc = 0
        elif key == 'x_sepas': 
            inc = 0
        elif key == 'edge_index_sepa_cols':  
            inc = torch.tensor([[N_SEPAS],
                                [self.x_cols.size(0)]])
        elif key == 'edge_vals_sepa_cols': 
            inc = 0
        elif key == 'edge_index_sepa_rows': 
            inc = torch.tensor([[N_SEPAS],
                                [self.x_rows.size(0)]])
        elif key == 'edge_vals_sepa_rows':  
            inc = 0
        elif key == 'edge_index_sepa_self':  
            inc = torch.tensor([[N_SEPAS],
                                [N_SEPAS]])
        elif key == 'edge_vals_sepa_self': 
            inc = 0
        elif key == 'edge_index_cuts':
            inc = torch.tensor([
                [self.x_cuts.size(0)],
                [self.x_cols.size(0)]])
        elif key == 'edge_vals_cuts':
            inc = 0
        elif key == 'edge_index_rows':
            inc = torch.tensor([
                [self.x_cuts.size(0)],
                [self.x_rows.size(0)]])
        elif key == 'edge_vals_rows':
            inc = 0
        elif key == 'edge_index_self':
            inc = torch.tensor([
                [self.x_cuts.size(0)],
                [self.x_cuts.size(0)]])
        elif key == 'edge_vals_self':
            inc = 0
        elif key == 'edge_index_rowcols':
            inc = torch.tensor([
                [self.x_rows.size(0)],
                [self.x_cols.size(0)]])
        elif key == 'edge_vals_rowcols':
            inc = 0
        else:
            # print('Resorting to default')
            inc = super().__inc__(key, value, *args, **kwargs)
        return inc

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'x_cuts':
            cat_dim = 0
        elif key == 'x_rows':
            cat_dim = 0
        elif key == 'x_cols':
            cat_dim = 0
        elif key == 'edge_index_cuts':
            cat_dim = 1
        elif key == 'edge_vals_cuts':
            cat_dim = 0
        elif key == 'edge_index_rows':
            cat_dim = 1
        elif key == 'edge_vals_rows':
            cat_dim = 0
        elif key == 'edge_index_self':
            cat_dim = 1
        elif key == 'edge_vals_self':
            cat_dim = 0
        elif key == 'edge_index_rowcols':
            cat_dim = 1
        elif key == 'edge_vals_rowcols':
            cat_dim = 0
        elif key == 'lpvals':
            cat_dim = 0
        elif key == 'sepa_settings':  
            cat_dim = 0
        elif key == 'masks':  
            cat_dim = 0
        elif key == 'x_sepas': 
            cat_dim = 0
        elif key == 'edge_index_sepa_cols':  
            cat_dim = 1
        elif key == 'edge_vals_sepa_cols':
            cat_dim = 0
        elif key == 'edge_index_sepa_rows': 
            cat_dim = 1
        elif key == 'edge_vals_sepa_rows':  
            cat_dim = 0
        elif key == 'edge_index_sepa_self':  
            cat_dim = 1
        elif key == 'edge_vals_sepa_self': 
            cat_dim = 0
        else:
            # print('Resorting to default')
            cat_dim = super().__cat_dim__(key, value, *args, **kwargs)
        return cat_dim

def getDataloaders(train_set, args, batch_size=64, shuffle_flag=True):
    num_workers = args.n_cpus
    pin_memory=False
    follow_batch = ['x_sepas', 'x_rows', 'x_cols']

    trainloader = torch_geometric.loader.DataLoader(
        bandit_dataset(train_set), 
        batch_size=batch_size, 
        shuffle=shuffle_flag,
        follow_batch=follow_batch, 
        num_workers=0, 
        pin_memory=pin_memory
    )

    return trainloader
