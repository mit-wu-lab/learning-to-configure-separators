from abc import ABC, abstractmethod
import numpy as np
import torch
from torch_geometric.data import Data
import json
import numpy as np
import os

import utils as _utils

X_CUTS_DIM = 26
LPVALS_DIM = 2

ROW_DIM = 39 + 1 + 2 #   prev 39 #   prev 39 + 1 lol
CUT_DIM = 39 + 1 + 2 #   prev 39 #   prev 39 + 1 lol
EDGE_DIM_CUTS = 2
EDGE_DIM_ROWS = 1
COL_DIM = 17 + 1  #   prev 17
N_SEPAS = 17  #  
SEPA_DIM = 22 + 1  #  
EDGE_DIM_SEPAS = 1  #  


class MyData(Data):
    ###  
    x_cuts_dim = X_CUTS_DIM
    lpvals_dim = LPVALS_DIM

    row_dim = ROW_DIM
    cut_dim = CUT_DIM
    edge_dim_cuts = EDGE_DIM_CUTS
    edge_dim_rows = EDGE_DIM_ROWS
    col_dim = COL_DIM
    sepa_dim = SEPA_DIM  #  
    edge_dim_sepas = EDGE_DIM_SEPAS  #  
    ###

    def __init__(
            self,
            x_cuts,
            x_rows,
            x_cols,
            edge_index_cuts, #
            edge_vals_cuts,
            edge_index_rows,
            edge_vals_rows,
            edge_index_self,
            edge_vals_self,
            edge_index_rowcols,
            edge_vals_rowcols,
            lpvals,
            sepa_settings,  #  
            masks,   #  
            x_sepas, edge_index_sepa_cols, edge_vals_sepa_cols,
            edge_index_sepa_rows, edge_vals_sepa_rows, edge_index_sepa_self, edge_vals_sepa_self):  #  

        super().__init__()
        self.x_cuts = x_cuts
        self.x_rows = x_rows
        self.x_cols = x_cols

        self.edge_index_cuts = edge_index_cuts
        self.edge_vals_cuts = edge_vals_cuts

        self.edge_index_rows = edge_index_rows
        self.edge_vals_rows = edge_vals_rows

        self.edge_index_self = edge_index_self
        self.edge_vals_self = edge_vals_self

        self.edge_index_rowcols = edge_index_rowcols
        self.edge_vals_rowcols = edge_vals_rowcols

        self.lpvals = lpvals  # first col is old_lp, second col is new_lp

        self.sepa_settings = sepa_settings  #  
        self.masks = masks  #  

        self.x_sepas = x_sepas  #  
        self.edge_index_sepa_cols = edge_index_sepa_cols  #  
        self.edge_vals_sepa_cols = edge_vals_sepa_cols  #  
        self.edge_index_sepa_rows = edge_index_sepa_rows  #  
        self.edge_vals_sepa_rows = edge_vals_sepa_rows  #  
        self.edge_index_sepa_self = edge_index_sepa_self  #  
        self.edge_vals_sepa_self = edge_vals_sepa_self  #  

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'x_cuts':
            inc = 0
        elif key == 'x_rows':
            inc = 0
        elif key == 'x_cols':
            inc = 0
        elif key == 'lpvals':
            inc = 0
        elif key == 'sepa_settings':  #  
            inc = 0
        elif key == 'masks':  #  
            inc = 0
        elif key == 'x_sepas':  #  
            inc = 0
        elif key == 'edge_index_sepa_cols':  #  
            inc = torch.tensor([[N_SEPAS],
                                [self.x_cols.size(0)]])
        elif key == 'edge_vals_sepa_cols':  #  
            inc = 0
        elif key == 'edge_index_sepa_rows':  #  
            inc = torch.tensor([[N_SEPAS],
                                [self.x_rows.size(0)]])
        elif key == 'edge_vals_sepa_rows':  #  
            inc = 0
        elif key == 'edge_index_sepa_self':  #  
            inc = torch.tensor([[N_SEPAS],
                                [N_SEPAS]])
        elif key == 'edge_vals_sepa_self':  #  
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
            print('Resorting to default')
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
        elif key == 'sepa_settings':  #  
            cat_dim = 0
        elif key == 'masks':  #  
            cat_dim = 0
        elif key == 'x_sepas':  #  
            cat_dim = 0
        elif key == 'edge_index_sepa_cols':  #  
            cat_dim = 1
        elif key == 'edge_vals_sepa_cols':  #  
            cat_dim = 0
        elif key == 'edge_index_sepa_rows':  #  
            cat_dim = 1
        elif key == 'edge_vals_sepa_rows':  #  
            cat_dim = 0
        elif key == 'edge_index_sepa_self':  #  
            cat_dim = 1
        elif key == 'edge_vals_sepa_self':  #  
            cat_dim = 0
        else:
            print('Resorting to default')
            cat_dim = super().__cat_dim__(key, value, *args, **kwargs)
        return cat_dim

    @classmethod
    def from_path(cls, path):
        raw_data = cls.load_rawdata(path)
        min_parall = cls.get_minparallelism(path)
        maxnum_cutixs, maxnum_rowixs, maxnum_rowcolixs = cls.get_maxnums(path)
        processed_data = cls.from_rawdata(raw_data, min_parall, maxnum_cutixs, maxnum_rowixs, maxnum_rowcolixs)
        self = cls(*processed_data)
        return self

    @classmethod
    def get_minparallelism(cls, path):
        if 'binpacking-66-66' in path:
            min_parall = 0.9
        elif 'packing-60-60' in path:
            min_parall = 0.85
        elif 'maxcut-14-40' in path:
            min_parall = 0.0
        elif 'planning-40' in path:
            min_parall = 0.0
        else:
            min_parall = 0.0
            # raise ValueError('What is minparallelism?')
        return min_parall

    @classmethod
    def get_maxnums(cls, path):
        # for cut <-> col
        if 'nn' in path:
            maxnum_cutixs = 200
            maxnum_rowixs = 200
            maxnum_rowcolixs = 200
        else:
            maxnum_cutixs, maxnum_rowixs, maxnum_rowcolixs = 1e6, 1e6, 1e6
            # raise ValueError('What is maxnum_cuixs?')
        return maxnum_cutixs, maxnum_rowixs, maxnum_rowcolixs


    @classmethod
    def load_rawdata(cls, path):
        raw_data = {}
        for file in [
                'cut_input_scores.npy',
                'row_input_scores.npy',
                'cut_lookahead_scores.npy',
                'cutrow_parallelism.npy',
                'cut_parallelism.npy',
                'masks.npy']:  #  

            arr = _utils.load_numpy(os.path.join(path, file))
            raw_data[file] = arr

        for file in [
                'cut_features.pkl',
                'row_features.pkl',
                'col_features.pkl',
                'cut_coefs.pkl',
                'row_coefs.pkl',
                'sepa_settings.pkl',  #  
                'sepa_features.pkl']:  #  

            feat_dicts = _utils.load_pickle(os.path.join(path, file))
            raw_data[file] = feat_dicts

        return raw_data

    @classmethod
    def from_rawdata(cls, raw_data, min_parall, maxnum_cutixs=1e6, maxnum_rowixs=1e6, maxnum_rowcolixs=1e6, args=None):

        # Make x_cuts
        ### cut scores
        scores = torch.FloatTensor(raw_data['cut_input_scores.npy'])
        relative_scores = (scores - scores.mean(0) ) / (scores.std(0) + 1e-5)
        relative_scores[~torch.isfinite(relative_scores)] = -1.0  # overwrite

        ## cut features
        x_cuts = cls.make_xcuts(
            raw_data['cut_features.pkl'],
            raw_data['cut_input_scores.npy'])

        x_rows = cls.make_xrows(
            raw_data['row_features.pkl'],
            raw_data['row_input_scores.npy'],
        )

        x_cols = cls.make_xcols(
            raw_data['col_features.pkl'],
        )

        edge_index_cuts, edge_vals_cuts = cls.make_edge_cols(
            raw_data['cut_coefs.pkl'], maxnum_cutixs
        )

        edge_index_rows, edge_vals_rows = cls.make_edge_rows(
            raw_data['cutrow_parallelism.npy'],
            min_parall, maxnum_rowixs
        )

        edge_index_self = cls.make_edge_index_self(x_cuts.size(0))
        edge_vals_self = cls.make_edge_vals_self(
            raw_data['cut_parallelism.npy']
        )

        edge_index_rowcols, edge_vals_rowcols = cls.make_edge_cols(
            raw_data['row_coefs.pkl'], maxnum_rowcolixs
        )
        # lpvals
        lpvals = torch.empty( size=(len(x_cuts), 2) )
        lpvals[:, 0] = torch.FloatTensor(raw_data['cut_lookahead_scores.npy'][:, 0])
        lpvals[:, 1] = torch.FloatTensor(raw_data['cut_lookahead_scores.npy'][:, 2])

        sepa_settings = torch.FloatTensor(raw_data['sepa_settings.pkl'])   #  
        masks = torch.BoolTensor(raw_data['masks.npy'])   #  

        #  
        x_sepas = cls.make_xsepas(
            raw_data['sepa_features.pkl'],
            args
        )

        #  
        edge_index_sepa_cols, edge_vals_sepa_cols = cls.make_edge_sepas_cols(
            N_SEPAS, x_cols.size(0)
        )

        #  
        edge_index_sepa_rows, edge_vals_sepa_rows = cls.make_edge_sepa_rows(
            N_SEPAS, x_rows.size(0)
        )

        #  
        edge_index_sepa_self, edge_vals_sepa_self = cls.make_edge_sepa_self(N_SEPAS)

        if torch.any(torch.isinf(x_rows)):
            print(torch.where(torch.isinf(x_rows)))

        #  
        return (x_cuts, x_rows, x_cols, edge_index_cuts, edge_vals_cuts, edge_index_rows, edge_vals_rows,
                edge_index_self, edge_vals_self, edge_index_rowcols, edge_vals_rowcols, lpvals,
                sepa_settings, masks, x_sepas, edge_index_sepa_cols, edge_vals_sepa_cols,
                edge_index_sepa_rows, edge_vals_sepa_rows, edge_index_sepa_self, edge_vals_sepa_self)

    @classmethod
    def make_xcuts(cls, features, scores):

        vecs = []
        for feat_dict in features:
            vec = cls.get_cut_vec(feat_dict)
            vecs.append(vec)

        vecs = np.array(vecs)
        if len(vecs) == 0:
            x = vecs.reshape(0, CUT_DIM)
            return torch.FloatTensor(x)
        x = torch.FloatTensor(np.concatenate([vecs, scores], axis=-1))
        x[:, 22][torch.isinf(x[:, 22])] = -1.0  # if we want to add a new cut type, need to update from 20 to 21 #  add 2
        return x

    @classmethod
    def make_xrows(cls, features, scores):

        vecs = []
        for feat_dict in features:
            vec = cls.get_row_vec(feat_dict)
            vecs.append(vec)

        vecs = np.array(vecs)

        # relative normalization of these scores
        # import pdb; pdb.set_trace()
        try:
            scores[:, -8] = np.clip( (scores[:, -8] - np.mean(scores[:, -8])) / (np.std(scores[:, -8]) + 1e-5), a_min = -10, a_max=10)
            scores[:, -3] = np.clip( (scores[:, -3] - np.mean(scores[:, -3])) / (np.std(scores[:, -3]) + 1e-5), a_min = -10, a_max=10)
        except:
            import pdb; pdb.set_trace()

        x = torch.FloatTensor(np.concatenate([vecs, scores], axis=-1))
        # import pdb; pdb.set_trace()
        # this feature is inf for one row often.
        x[:, 22][torch.isinf(x[:, 22])] = -1.0  # if we want to add a new cut type, need to update from 20 to 21
        # import pdb; pdb.set_trace()
        return x

    @classmethod
    def make_xcols(cls, features):

        vecs = []
        for feat_dict in features:
            vec = cls.get_col_vec(feat_dict)
            vecs.append(vec)

        x = torch.FloatTensor(np.array(vecs))

        return x

    @classmethod
    def make_edge_cols(cls, coefs, maxnum=1e6):

        edge_ixs_top = []
        edge_ixs_bottom = []
        edge_vals_raw = []
        edge_vals_norm = []

        for cut_ix, (col_ix, vals) in coefs.items():
            edge_ixs_top.append( np.ones(len(col_ix)) * cut_ix )
            edge_ixs_bottom.append( np.array(col_ix) )

            vals = np.array(vals)
            edge_vals_raw.append( vals )
            edge_vals_norm.append( vals / np.linalg.norm(vals))

        if len(edge_ixs_top) == 0:  #  
            edge_ixs = np.stack([[], []])
            edge_vals = np.stack([[], []])
        else:
            edge_ixs = np.stack([
                np.concatenate(edge_ixs_top),
                np.concatenate(edge_ixs_bottom)])
            edge_vals = np.stack([
                np.concatenate(edge_vals_raw),
                np.concatenate(edge_vals_norm)])

        edge_ixs = torch.LongTensor(edge_ixs)
        edge_vals = torch.FloatTensor(edge_vals.T)

        # Filter..
        if maxnum < 1e6:  # nnv
            if len(edge_vals) > maxnum:
                threshold = torch.sort(edge_vals[:, 0], descending=True).values[maxnum]
                mask = (edge_vals[:, 0] > threshold)
                edge_vals = torch.stack( [
                    torch.masked_select(edge_vals[:,0], mask),
                    torch.masked_select(edge_vals[:,1], mask),
                    ], dim=-1)
                edge_ixs = torch.stack( [
                    torch.masked_select(edge_ixs[0, :], mask),
                    torch.masked_select(edge_ixs[1, :], mask),
                    ], dim=0)
            else:
                pass

        return edge_ixs, edge_vals

    @classmethod
    def make_edge_rows(cls, parallelism, min_parall, maxnum_rowixs=1e6):
        # cut ix is top, row ix is bottom

        if maxnum_rowixs < 1e6:  # nnv
            max_edges = (parallelism.shape[0] * parallelism.shape[1])
            if max_edges < maxnum_rowixs:
                min_parall = 0.0
            else:
                min_parall = np.sort(parallelism.flatten())[::-1][maxnum_rowixs]

        edge_ixs = np.where(parallelism > min_parall)
        if len(edge_ixs[0]) == 0:  #  
            # print('No edge ixs.')
            edge_ixs = np.stack([[], []])
            edge_vals = np.stack([[]])
            edge_ixs = torch.LongTensor(edge_ixs)
            edge_vals = torch.FloatTensor(edge_vals.T)
            return edge_ixs, edge_vals

        edge_vals = parallelism[edge_ixs][:, None]

        edge_ixs = torch.LongTensor(np.stack(edge_ixs))
        edge_vals = torch.FloatTensor(edge_vals)

        # Clip..
        edge_vals = torch.clamp(edge_vals, max=2.0) # max = 2.0 so have indication for wrong value..

        return edge_ixs, edge_vals

    @classmethod
    def make_edge_index_self(cls, n_cuts):
        # no self-loops..
        _temp = np.triu_indices(n_cuts - 1)
        triu_ixs = np.stack((_temp[0], _temp[1] + 1))
        rev = np.stack([triu_ixs[1, :], triu_ixs[0, :]])
        edge_ixs = np.concatenate((triu_ixs, rev), axis=-1) # no self-loops
        edge_ixs = torch.LongTensor(edge_ixs)
        
        return edge_ixs

    @classmethod
    def make_edge_vals_self(cls, parallelism):
        vals = torch.FloatTensor(parallelism)
        vals = vals.repeat(2).unsqueeze(1)
        return vals

    @classmethod
    def get_cut_vec(cls, feat_dict):
        vec = []
        # Type
        vec.append(1.0 if feat_dict['origin_type'] == 0 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 1 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 2 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 3 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 4 else 0.0 )
        # Cut Type
        vec.append( 1.0 if 'cmir' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'flowcover' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'clique' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'dis' in feat_dict['rname'] else 0.0 ) # ?
        vec.append( 1.0 if 'gom' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'implbd' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'mcf' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'oddcycle' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'scg' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'zerohalf' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'cgcut' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'obj' in feat_dict['rname'] else 0.0 )

        # basis status
        vec.append( 1.0 if feat_dict['basisstatus'] == 0 else 0.0 )  # basestat one-hot {lower: 0, basic: 1, upper: 2, zero: 3}
        vec.append( 1.0 if feat_dict['basisstatus'] == 1 else 0.0 )
        vec.append( 1.0 if feat_dict['basisstatus'] == 2 else 0.0 )
        vec.append( 1.0 if feat_dict['basisstatus'] == 3 else 0.0 )
        # other info
        vec.append( feat_dict['rank'] )
        # normalization (following Giulia)
        lhs = feat_dict['lhs']
        rhs = feat_dict['rhs']
        cst = feat_dict['rhs']
        nlps = feat_dict['nlps']
        cste = feat_dict['cste']

        activity = feat_dict['activity']
        row_norm = feat_dict['row_norm']
        obj_norm = feat_dict['obj_norm']
        dualsol = feat_dict['dualsol']

        unshifted_lhs = None if np.isinf(lhs) else lhs - cst
        unshifted_rhs = None if np.isinf(rhs) else rhs - cst

        if unshifted_lhs is not None:
            bias = -1. * unshifted_lhs / row_norm
            dualsol = -1. *  dualsol / (row_norm * obj_norm)
        if unshifted_rhs is not None:
            bias = unshifted_rhs / row_norm
            dualsol = dualsol / (row_norm * obj_norm)
        # values
        vec.append( bias )
        vec.append( dualsol )
        vec.append( 1.0 if np.isclose(activity, lhs) else 0.0 ) # at_lhs
        vec.append( 1.0 if np.isclose(activity, rhs) else 0.0 ) # at_rhs
        vec.append( feat_dict['nlpnonz'] / feat_dict['ncols'] )
        vec.append( feat_dict['age'] / (feat_dict['nlps'] + feat_dict['cste']) )
        vec.append( feat_dict['nlpsaftercreation'] / (feat_dict['nlps'] + feat_dict['cste']) )
        vec.append( feat_dict['intcols'] / feat_dict['ncols'] )
        # flags
        vec.append( 1.0 if feat_dict['is_integral'] else 0.0 )
        vec.append( 1.0 if feat_dict['is_removable'] else 0.0 )  # could be removed if we have binary identifier for cuts
        vec.append( 1.0 if feat_dict['is_in_lp'] else 0.0 )  # could be removed if we have binary identifier for cuts

        #round feature
        vec.append(feat_dict['round_num'] if 'round_num' in feat_dict else 0)

        return vec

    @classmethod
    def get_row_vec(cls, feat_dict):
        vec = []
        # Type
        vec.append(1.0 if feat_dict['origin_type'] == 0 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 1 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 2 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 3 else 0.0 )
        vec.append(1.0 if feat_dict['origin_type'] == 4 else 0.0 )
        # Cut Type
        vec.append( 1.0 if 'cmir' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'flowcover' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'clique' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'dis' in feat_dict['rname'] else 0.0 ) # ?
        vec.append( 1.0 if 'gom' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'implbd' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'mcf' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'oddcycle' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'scg' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'zerohalf' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'cgcut' in feat_dict['rname'] else 0.0 )
        vec.append( 1.0 if 'objrow' in feat_dict['rname'] else 0.0 ) # intobj ?

        # basis status
        vec.append( 1.0 if feat_dict['basisstatus'] == 0 else 0.0 )  # basestat one-hot {lower: 0, basic: 1, upper: 2, zero: 3}
        vec.append( 1.0 if feat_dict['basisstatus'] == 1 else 0.0 )
        vec.append( 1.0 if feat_dict['basisstatus'] == 2 else 0.0 )
        vec.append( 1.0 if feat_dict['basisstatus'] == 3 else 0.0 )
        # other info
        vec.append( feat_dict['rank'] )
        # normalization (following Giulia)
        lhs = feat_dict['lhs']
        rhs = feat_dict['rhs']
        cst = feat_dict['rhs']
        nlps = feat_dict['nlps']
        cste = feat_dict['cste']

        activity = feat_dict['activity']
        row_norm = feat_dict['row_norm']
        obj_norm = feat_dict['obj_norm']
        dualsol = feat_dict['dualsol']

        unshifted_lhs = None if np.isinf(lhs) else lhs - cst
        unshifted_rhs = None if np.isinf(rhs) else rhs - cst

        if unshifted_lhs is not None:
            bias = -1. * unshifted_lhs / row_norm
            dualsol = -1. *  dualsol / (row_norm * obj_norm)
        if unshifted_rhs is not None:
            bias = unshifted_rhs / row_norm
            dualsol = dualsol / (row_norm * obj_norm)
        # values
        vec.append( bias )
        vec.append( dualsol )
        vec.append( 1.0 if np.isclose(activity, lhs) else 0.0 ) # at_lhs
        vec.append( 1.0 if np.isclose(activity, rhs) else 0.0 ) # at_rhs
        vec.append( feat_dict['nlpnonz'] / feat_dict['ncols'] )
        vec.append( feat_dict['age'] / (feat_dict['nlps'] + feat_dict['cste']) )
        vec.append( feat_dict['nlpsaftercreation'] / (feat_dict['nlps'] + feat_dict['cste']) )
        vec.append( feat_dict['intcols'] / feat_dict['ncols'] )

        # flags
        vec.append( 1.0 if feat_dict['is_integral'] else 0.0 )
        vec.append( 1.0 if feat_dict['is_removable'] else 0.0 )  # could be removed if we have binary identifier for cuts
        vec.append( 1.0 if feat_dict['is_in_lp'] else 0.0 )  # could be removed if we have binary identifier for cuts

        #round feature
        vec.append(feat_dict['round_num'] if 'round_num' in feat_dict else 0)

        return vec

    @classmethod
    def get_col_vec(cls, feat_dict):
        vec = []
        # type
        vec.append( 1.0 if feat_dict['type'] == 0 else 0.0 ) # binary
        vec.append( 1.0 if feat_dict['type'] == 1 else 0.0 ) # integer
        vec.append( 1.0 if feat_dict['type'] == 2 else 0.0 ) # implicit-int
        vec.append( 1.0 if feat_dict['type'] == 3 else 0.0 ) # continuous
        # bounds
        vec.append( 1.0 if feat_dict['lb'] is not None else 0.0 ) # has_lower_bound
        vec.append( 1.0 if feat_dict['ub'] is not None else 0.0 ) # has_upper_bound
        # basis status
        vec.append( 1.0 if feat_dict['basestat'] == 0 else 0.0 )
        vec.append( 1.0 if feat_dict['basestat'] == 1 else 0.0 )
        vec.append( 1.0 if feat_dict['basestat'] == 2 else 0.0 )
        vec.append( 1.0 if feat_dict['basestat'] == 3 else 0.0 )
        # values
        vec.append( feat_dict['norm_coef'])
        vec.append( feat_dict['norm_redcost'] ) # was already normalized :()
        vec.append( feat_dict['norm_age'] ) # war already normalized :()
        vec.append( feat_dict['solval'])
        vec.append( feat_dict['solfrac'])
        vec.append( 1.0 if feat_dict['sol_is_at_lb'] else 0.0 )
        vec.append( 1.0 if feat_dict['sol_is_at_ub'] else 0.0 )

        # round feature
        vec.append(feat_dict['round_num'] if 'round_num' in feat_dict else 0)

        return vec

    #  
    @classmethod
    def get_sepa_vec(cls, feat_dict, args):
        vec = []
        # Cut Type
        vec.append(1.0 if 'disjunctive' in feat_dict['name'] else 0.0) 
        vec.append(1.0 if 'convexproj' in feat_dict['name'] else 0.0 )
        vec.append(1.0 if 'gauge' in feat_dict['name'] else 0.0 )
        vec.append(1.0 if 'impliedbounds' in feat_dict['name'] else 0.0)
        vec.append(1.0 if 'intobj' in feat_dict['name'] else 0.0 )
        vec.append(1.0 if 'gomory' in feat_dict['name'] else 0.0)
        vec.append(1.0 if 'cgmip' in feat_dict['name'] else 0.0 ) 
        vec.append(1.0 if 'strongcg' in feat_dict['name'] else 0.0)
        vec.append(1.0 if 'aggregation' in feat_dict['name'] else 0.0 )
        vec.append(1.0 if 'clique' in feat_dict['name'] else 0.0)
        vec.append(1.0 if 'zerohalf' in feat_dict['name'] else 0.0)
        vec.append(1.0 if 'mcf' in feat_dict['name'] else 0.0)
        vec.append(1.0 if 'eccuts' in feat_dict['name'] else 0.0 )
        vec.append(1.0 if 'oddcycle' in feat_dict['name'] else 0.0)
        vec.append(1.0 if 'flowcover' in feat_dict['name'] else 0.0)
        vec.append(1.0 if 'cmir' in feat_dict['name'] else 0.0)
        vec.append(1.0 if 'rapidlearning' in feat_dict['name'] else 0.0 )
            
        return vec

    #  
    @classmethod
    def make_xsepas(cls, features, args):

        vecs = []
        for feat_dict in features:
            vec = cls.get_sepa_vec(feat_dict, args)
            vecs.append(vec)

        x = torch.FloatTensor(np.array(vecs))

        return x


    #  
    @classmethod
    def make_edge_sepas_cols(cls, n_sepas, n_cols):
        # top: sepa idx; bottom: col idx -> fully connected
        edge_ixs_top = []
        edge_ixs_bottom = []
        edge_vals_raw = []
        edge_vals_norm = []

        for sepa_ix in range(n_sepas):
            edge_ixs_top.append(np.ones(n_cols) * sepa_ix)
            edge_ixs_bottom.append(np.arange(n_cols))
            edge_vals_raw.append(np.ones(n_cols))
            edge_vals_norm.append(np.ones(n_cols))

        if len(edge_ixs_top) == 0:  #  
            edge_ixs = np.stack([[], []])
            edge_vals = np.stack([[], []])
        else:
            edge_ixs = np.stack([
                np.concatenate(edge_ixs_top),
                np.concatenate(edge_ixs_bottom)])
            edge_vals = np.stack([
                np.concatenate(edge_vals_raw),
                ])  # np.concatenate(edge_vals_norm)

        edge_ixs = torch.LongTensor(edge_ixs)
        edge_vals = torch.FloatTensor(edge_vals.T)

        return edge_ixs, edge_vals

    #  
    @classmethod
    def make_edge_sepa_rows(cls, n_sepas, n_rows):
        # sepa ix is top, row ix is bottom
        edge_ixs_top = []
        edge_ixs_bottom = []
        edge_vals_raw = []
        edge_vals_norm = []

        for sepa_ix in range(n_sepas):
            edge_ixs_top.append(np.ones(n_rows) * sepa_ix)
            edge_ixs_bottom.append(np.arange(n_rows))
            edge_vals_raw.append(np.ones(n_rows))
            edge_vals_norm.append(np.ones(n_rows))

        if len(edge_ixs_top) == 0:  #  
            edge_ixs = np.stack([[], []])
            edge_vals = np.stack([[], []])
        else:
            edge_ixs = np.stack([
                np.concatenate(edge_ixs_top),
                np.concatenate(edge_ixs_bottom)])
            edge_vals = np.stack([
                np.concatenate(edge_vals_raw),
                ])  # np.concatenate(edge_vals_norm)

        edge_ixs = torch.LongTensor(edge_ixs)
        edge_vals = torch.FloatTensor(edge_vals.T)

        return edge_ixs, edge_vals

    #  
    @classmethod
    def make_edge_sepa_self(cls, n_sepas):
        # no self-loops..
        _temp = np.triu_indices(n_sepas - 1)
        triu_ixs = np.stack((_temp[0], _temp[1] + 1))
        rev = np.stack([triu_ixs[1, :], triu_ixs[0, :]])
        edge_ixs = np.concatenate((triu_ixs, rev), axis=-1) # no self-loops
        edge_ixs = torch.LongTensor(edge_ixs)

        edge_vals = torch.ones((edge_ixs.size(-1), 1))
        return edge_ixs, edge_vals
