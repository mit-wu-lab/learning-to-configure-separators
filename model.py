import torch
import torch.nn as nn
import torch_geometric
import data as _data
import context as _context
import numpy as np

def getModel(modelstr="Neural_UCB"):
    if modelstr == 'Neural_UCB':
        return Neural_UCB()
    else:
        raise ValueError('Unknown model.')

# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """
    def __init__(self, edge_dim):
        super().__init__('add')
        emb_size = 64
        self.edge_dim = edge_dim

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):

        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_edge(edge_features)
                                           + self.feature_module_right(node_features_j))
        return output

class Neural_UCB(torch.nn.Module):
    emb_size = 64
    row_dim = _data.MyData.row_dim
    edge_dim_cuts = _data.MyData.edge_dim_cuts
    edge_dim_rows = _data.MyData.edge_dim_rows
    col_dim = _data.MyData.col_dim
    num_sepa = 17  # 10
    sepa_dim = 1 
    edge_dim_sepas = _data.MyData.edge_dim_sepas 

    def __init__(self):
        super().__init__()
        # ROW EMBEDDING
        self.row_embedding = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.row_dim),
            torch.nn.Linear(self.row_dim, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )

        self.sepa_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(self.sepa_dim),
            torch.nn.Linear(self.sepa_dim, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_norm_sepas = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.edge_dim_sepas),
        )

        self.edge_norm_rowcols = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.edge_dim_cuts),
        )

        self.edge_norm_rows = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.edge_dim_rows),
        )

        # colIABLE EMBEDDING
        self.col_embedding = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.col_dim),
            torch.nn.Linear(self.col_dim, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )

        self.conv_col_to_row = BipartiteGraphConvolution(edge_dim=2)
        self.conv_row_to_col = BipartiteGraphConvolution(edge_dim=2)

        self.conv_sepa_to_col = BipartiteGraphConvolution(edge_dim=1)
        self.conv_col_to_sepa = BipartiteGraphConvolution(edge_dim=1)

        self.conv_sepa_to_row = BipartiteGraphConvolution(edge_dim=1)
        self.conv_row_to_sepa = BipartiteGraphConvolution(edge_dim=1)

        self.transformer_conv = torch_geometric.nn.conv.TransformerConv(
            in_channels = self.emb_size,
            out_channels = self.emb_size // 4,
            heads=4,
            concat=True,
            dropout=0.1,
            edge_dim=1,
        )

        self.sepa_output_embed_module = torch.nn.Sequential(
            torch.nn.Linear(2*self.emb_size+1, self.emb_size),
            torch.nn.ReLU(),
        )
        
        self.row_output_embed_module = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )
        
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size*3, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, 1),
            torch.nn.Sigmoid(),
        )

    def get_device(self):
        return self.row_embedding[1].weight.device

    def load_grad_dict(self, grad_dict):
        state_dict = self.state_dict()
        assert isinstance(grad_dict, type(state_dict))

        for (name, param) in self.named_parameters():
            if name in grad_dict:

                grad = grad_dict[name]
                param.grad = grad
            else:
                param.grad = torch.zeros_like(param.data)

    def forward(
            self,
            inp,
        ):
        device = self.get_device()
        x_rows = inp.x_rows.to(device)
        x_cols = inp.x_cols.to(device)
        x_sepas = inp.x_sepas.to(device)

        edge_index_rowcols = inp.edge_index_rowcols.to(device)  # row ix is top, col ix is bottom
        edge_vals_rowcols = inp.edge_vals_rowcols.to(device)  # currently fully connected, all 1 weight

        edge_index_sepa_cols = inp.edge_index_sepa_cols.to(device)  # sepa ix is top, col ix is bottom
        edge_vals_sepa_cols = inp.edge_vals_sepa_cols.to(device)  # currently fully connected, all 1 weight

        edge_index_sepa_rows = inp.edge_index_sepa_rows.to(device)  # sepa ix is top, row ix is bottom
        edge_vals_sepa_rows = inp.edge_vals_sepa_rows.to(device)  # currently fully connected, all 1 weight

        edge_index_sepa_self = inp.edge_index_sepa_self.to(device)  # sepa and sepa self edge
        edge_vals_sepa_self = inp.edge_vals_sepa_self.to(device)  # currently fully connected, all 1 weight

        if hasattr(inp, 'x_sepas_batch'):
            x_sepas_batch = inp.x_sepas_batch.to(device)
            batch_size = inp.num_graphs
        else:
            x_sepas_batch = torch.zeros(x_sepas.shape[0], dtype=torch.long).to(device)
            batch_size = 1
        
        if hasattr(inp, 'x_cols_batch'):
            x_cols_batch = inp.x_cols_batch.to(device)
            batch_size = inp.num_graphs
        else:
            x_cols_batch = torch.zeros(x_cols.shape[0], dtype=torch.long).to(device)
            batch_size = 1

        if hasattr(inp, 'x_rows_batch'):
            x_rows_batch = inp.x_rows_batch.to(device)
        else:
            x_rows_batch = torch.zeros(x_rows.shape[0], dtype=torch.long).to(device)

        r_edge_index_sepa_cols = torch.stack([edge_index_sepa_cols[1], edge_index_sepa_cols[0]], dim=0)
        r_edge_index_sepa_rows = torch.stack([edge_index_sepa_rows[1], edge_index_sepa_rows[0]], dim=0)
        r_edge_index_rowcols = torch.stack([edge_index_rowcols[1], edge_index_rowcols[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        row_embd = self.row_embedding(x_rows)
        sepa_embd = self.sepa_embedding(x_sepas)

        col_embd = self.col_embedding(x_cols)

        edge_embd_sepas = self.edge_norm_sepas(edge_vals_sepa_cols)
        edge_embd_sepa_rows = self.edge_norm_rows(edge_vals_sepa_rows)
        edge_embd_rowcols = self.edge_norm_rowcols(edge_vals_rowcols)

        row_embd = self.conv_col_to_row(col_embd, r_edge_index_rowcols, edge_embd_rowcols, row_embd)
        col_embd = self.conv_row_to_col(row_embd, edge_index_rowcols, edge_embd_rowcols, col_embd)

        sepa_embd = self.conv_col_to_sepa(col_embd, r_edge_index_sepa_cols, edge_embd_sepas, sepa_embd)

        row_embd = self.conv_sepa_to_row(sepa_embd, edge_index_sepa_rows, edge_embd_sepa_rows, row_embd)
        sepa_embd = self.conv_row_to_sepa(row_embd, r_edge_index_sepa_rows, edge_embd_sepa_rows, sepa_embd)

        sepa_att = self.transformer_conv(sepa_embd, edge_index_sepa_self, edge_vals_sepa_self)
        
        sepa_att = self.sepa_output_embed_module(torch.cat([sepa_embd, sepa_att, x_sepas], dim=-1))  # nodes * feature
        row_att = self.row_output_embed_module(row_embd)  # nodes * feature
        output = torch.cat([torch_geometric.nn.global_mean_pool(sepa_att, x_sepas_batch, size=batch_size),
            torch_geometric.nn.global_mean_pool(row_att, x_rows_batch, size=batch_size),
            torch_geometric.nn.global_mean_pool(col_embd, x_cols_batch, size=batch_size)],
            dim=-1
        )
        
        output = self.output_module(output)  # n_batch * n_cuts

        return output

class NeuralUCB:
    def __init__(self, modelstr, lamb, nu, actions, args):
        self.model = getModel(modelstr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.lamb = lamb
        self.total_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.U = lamb * torch.ones((self.total_param,))
        self.nu = nu
        self.actions = actions
        self.ucb_on = args.ucb_on
        self.ucb_val_on = args.ucb_val_on
        self.wrong_time = 0
    
    def getActions(self, input_context, num=1, eva=False):
        self.model.train(False)
        UCBs = []
        g_list = []
        actions_context = []
        for action in self.actions:
            action_context = _context.getActionContext("bandit", input_context, action)
            actions_context.append(action_context)
            score = self.model(action_context)
            UCB = score.item()
            self.model.zero_grad()
            if self.ucb_on and (not eva or self.ucb_val_on):
                score.backward(retain_graph=True)
                tmp = []
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    tmp.append(p.grad.flatten().detach())
                g = torch.cat(tmp)
                g_list.append(g)
                if (g*g).shape != self.U.shape:
                    self.U = self.lamb * torch.ones((g*g).shape)# .to(self.device)
                    self.wrong_time += 1
                sigma = torch.sqrt(torch.sum(self.lamb * self.nu * g * g / self.U))
                UCB += sigma.item()
            UCBs.append(UCB)
        
        UCBs = np.array(UCBs)
        if eva == True:
            UCBs[UCBs < UCBs.max()-1e-9] = -np.inf
            
        softmax = torch.nn.Softmax(dim=0)
        probs = softmax(torch.tensor(UCBs))
        dist = torch.distributions.categorical.Categorical(probs) 
        is_sample = np.zeros(len(self.actions))
        sampled_actions = []
        sampled_actions_context = []
        sampled_actions_score = []
        while num > 0:
            arm = dist.sample()
            if is_sample[arm] == 0:
                sampled_actions.append(self.actions[arm])
                sampled_actions_context.append(actions_context[arm])
                sampled_actions_score.append(UCBs[arm])
                is_sample[arm] = 1
                num -= 1
                if self.ucb_on and not eva:
                    self.U += g_list[arm] * g_list[arm]
        return sampled_actions, sampled_actions_context
    
    def getActionsScores(self, input_context):
        self.model.train(False)
        UCBs = []
        g_list = []
        actions_context = []
        for action in self.actions:
            action_context = _context.getActionContext("bandit", input_context, action)
            actions_context.append(action_context)
            score = self.model(action_context)
            UCB = score.item()
            self.model.zero_grad()
            if self.ucb_on and self.ucb_val_on:
                score.backward(retain_graph=True)
                tmp = []
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    tmp.append(p.grad.flatten().detach())
                g = torch.cat(tmp)
                g_list.append(g)
                if (g*g).shape != self.U.shape:
                    self.U = self.lamb * torch.ones((g*g).shape)# .to(self.device)
                    self.wrong_time += 1
                sigma = torch.sqrt(torch.sum(self.lamb * self.nu * g * g / self.U))
                UCB += sigma.item()
            UCBs.append(UCB)
        
        return UCBs