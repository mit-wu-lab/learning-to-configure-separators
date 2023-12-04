import argparse
import os

import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import utils as _utils
import torch
import torch.nn as nn

from joblib import Parallel, delayed
from multiprocessing import cpu_count as _cpu_count
import pyscipopt as pyopt
import numpy as np
import copy
import math

import context as _context
import dataset as _dataset
import model as _b_model
from torch.utils.tensorboard import SummaryWriter

writer = None
writer_train_len = 0
# parameter 
path_to_randomness_control_set = './SCIP_settings/randomness_control.set'
embsize = None
instances_list = []

class SepaManager_SM(pyopt.Sepa):
    # Defaults (shouldn't matter)
    SEPA_NAME = '#SM'
    SEPA_DESC = 'special sepa manager'
    SEPA_FREQ = 1
    SEPA_MAXBOUNDDIST = 1.0
    SEPA_USESSUBSCIP = False
    SEPA_DELAY = False
    SEPA_PRIORITY = 1

    # Info record
    def __init__(self):
        super().__init__()
        self.History_SM = []

    def sepaexeclp(self):
        if self.sepa_round in self.actions.keys():
            action = self.actions[self.sepa_round]
            for index in range(len(self.sepa_list)):
                on_or_off = 1
                if action[index, 0] == 0:
                    on_or_off = -1
                self.model.setParam(f'separating/{self.sepa_list[index]}/freq', on_or_off)
        self.sepa_round += 1
        return {'result': pyopt.SCIP_RESULT.DIDNOTRUN}

    def sepaexecsol(self):
        # actual behaviour is implemented in method self.main
        return {'result': pyopt.SCIP_RESULT.DIDNOTRUN}

    def addModel(self, model, actions):
        r'''
        Call self.addModel(model) instead of model.addSeparator(self, **kwargs)
        # max_sepa_round_norm is the constant used to normalize the round feature in the network. It is only used when
        # a network is used to select sepa-settings or we need to save_state for neural network input.
        '''
        self._check_inputs(model)
        model.includeSepa(
            self,
            self.SEPA_NAME,
            self.SEPA_DESC,
            self.SEPA_PRIORITY,
            self.SEPA_FREQ,
            self.SEPA_MAXBOUNDDIST,
            self.SEPA_USESSUBSCIP,
            self.SEPA_DELAY)
        self.sepa_list = [
            # 'closecuts',
            'disjunctive',
            # '#SM',
            # '#CS',
            'convexproj',
            'gauge',
            'impliedbounds',
            'intobj',
            'gomory',
            'cgmip',
            'strongcg',
            'aggregation',
            'clique',
            'zerohalf',
            'mcf',
            'eccuts',
            'oddcycle',
            'flowcover',
            'cmir',
            'rapidlearning'
        ]
        self.model.setParam(f'separating/#SM/freq', 1)
        self.model.setParam(f'separating/closecuts/freq', -1)
        self.sepa_round = 0
        self.actions = actions
        assert self.model == model  # PySCIPOpt sets that in pyopt.Sepa

    def get_sepa_round(self):
        return self.sepa_round

    def _check_inputs(self, model):
        assert isinstance(model, pyopt.Model)
        # this checks that all attributes to includeSepa are correctly specified.
        assert isinstance(self.SEPA_NAME, str)
        assert isinstance(self.SEPA_DESC, str)
        assert isinstance(self.SEPA_PRIORITY, int)
        assert isinstance(self.SEPA_FREQ, int)
        assert isinstance(self.SEPA_MAXBOUNDDIST, float)
        assert isinstance(self.SEPA_USESSUBSCIP, bool)
        assert isinstance(self.SEPA_DELAY, bool)


def solve(path_to_problem, actions):
    global args

    # get the solve time of SCIP
    SCIP_time = 0
    for i in range(args.reward_avg_time):
        model = pyopt.Model()
        model.hideOutput(1)
        model.readProblem(path_to_problem)
        model.readParams(path_to_randomness_control_set)
        sepa_manager_SM = SepaManager_SM()
        sepa_manager_SM.addModel(model, actions={})
        if args.instances == "load_balancing" or args.instances == "miplib":
            model.setParam("limits/gap", 0.1)
        model.optimize()
        SCIP_time += model.getSolvingTime() / args.reward_avg_time

    # get the solve time of the action
    our_time = 0
    for i in range(args.reward_avg_time):
        model = pyopt.Model()
        model.hideOutput(1)
        model.readProblem(path_to_problem)
        model.readParams(path_to_randomness_control_set)
        sepa_manager_SM = SepaManager_SM()
        sepa_manager_SM.addModel(model, actions=actions)
        model.setParam("limits/time", SCIP_time * 2.5)
        if args.instances == "load_balancing" or args.instances == "miplib":
            model.setParam("limits/gap", 0.1)
        model.optimize()
        our_time += model.getSolvingTime() / args.reward_avg_time

    # clipping
    clipping = 1.5

    improv_time = (SCIP_time - our_time) / SCIP_time
    reward = improv_time
    
    if reward < -clipping:
        reward = -clipping
    return improv_time, reward

def train(train_set, val_set, nn_model, args):
    nn_model.train(True)
    optimizer = torch.optim.Adam(
        nn_model.parameters(),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=args.lr_decay
    )
    loss_fuction = torch.nn.MSELoss()
    train_epoch = args.train_epoch
    train_loader = _dataset.getDataloaders(train_set[max(0,len(train_set)-args.max_train_set):], args)
    for epoch in range(train_epoch):
        global writer_train_len
        # log: val
        if epoch % args.val_loss_freq == 0:
            with torch.no_grad():
                val_predicts = []
                val_labels = []
                for item in val_set:
                    val_action_context, val_label = item
                    val_labels.append(val_label)
                    val_predict = nn_model(val_action_context)
                    val_predicts.append(val_predict)
                val_loss = loss_fuction(torch.tensor(val_labels).float(), torch.tensor(val_predicts).float())
                
                writer.add_scalar('val_loss', val_loss, writer_train_len)
                writer.add_scalar('val_avg_bias', math.sqrt(val_loss), writer_train_len)
                print(f"epoch: {epoch}, val loss: {val_loss}, val avg bias: {math.sqrt(val_loss)}")
    
        loss_avg = 0
        for _, action_contexts in enumerate(train_loader):
            num = action_contexts.num_graphs
            predicts = nn_model(action_contexts)
            labels = action_contexts.labels
            loss = loss_fuction(torch.tensor(labels).float(), predicts[:, 0])
            loss_avg += loss * num / len(train_set)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # log: train
        writer.add_scalar('train_loss', loss_avg, writer_train_len)
        writer.add_scalar('train_avg_bias', math.sqrt(loss_avg), writer_train_len)
        writer_train_len += 1
        print(f"epoch: {epoch}, loss: {loss_avg}, avg bias: {math.sqrt(loss_avg)}")
    nn_model.train(False)
    return nn_model

def get_path_to_problem(index, split='train'):
    instance_suffix = f"model-{index}/model.mps"
    if args.instances == "nnv":
        instance_suffix = f"model-{index}/model.proto.lp"
    elif args.instances == "load_balancing":
        instance_suffix = f"load_balancing_{index}.mps.gz"
    elif args.instances == "miplib":
        instance_suffix = instances_list[index]

    path_to_problem = os.path.join(
        args.path_to_data_dir, 
        args.instances+'_'+split, 
        instance_suffix
    )
    
    return path_to_problem

def multiprocess_context_helper(helper_args):
    process_num, Bandits, Bandit, val_flag = helper_args
    # sample problem
    print(f"process {process_num}: sampling the problem")
    model_num = None
    path_to_problem = None

    if val_flag == False:
        while True:
            global args
            model_num = np.random.randint(0, args.train_index_end)
            path_to_problem = get_path_to_problem(model_num, "train")
            if os.path.exists(path_to_problem):
                break
    else:
        model_num = process_num
        path_to_problem = get_path_to_problem(model_num, "val")
        
    # get context for the problem
    input_context, actions = _context.getInputContextAndInstruction(path_to_problem, args.step_k2, args.instances, Bandits=Bandits)
    if input_context == None:
        if val_flag == True:
            my_len = 1
        else:
            my_len = args.action_sampled_num
        return [-1] * my_len, \
               [-1] * my_len, \
               [-1] * my_len, \
               [-1] * my_len
               
    actions_sampled_num = 1
    if val_flag == False:
        actions_sampled_num = args.action_sampled_num
    actions_play, actions_context = Bandit.getActions(input_context, actions_sampled_num, val_flag)
    actions = [actions] * len(actions_play)
    for idx, action in enumerate(actions_play):
        actions[idx][args.step_k2] = action
        print("------------------------")
        for key in actions[idx].keys():
            print(f"step {key}, {actions[idx][key][:,0]}")
        print("------------------------")
    return actions_context, \
           actions, \
           [path_to_problem] * len(actions_play), \
           [process_num] * len(actions_play), \


def multiprocess_solve_helper(helper_args):
    subprocess_num, process_num, actions, path_to_problem = helper_args
    print(f"process {process_num}-{subprocess_num}: solving the problem with the chosen arm...")
    improv, reward = solve(path_to_problem, actions)
    print(f"process {process_num}-{subprocess_num}: at this iteration, the chosen arm improves {improv} r.s.t. SCIP, and get reward {reward}")
    return reward, improv


def saveModel(model, modelpath):
    assert isinstance(model, nn.Module)
    torch.save(model.state_dict(), modelpath)


def unpack_action_contexts(elements, num):
    A, B, C, D = [], [], [], []
    a, b, c, d = elements
    for i in range(num):
        if type(a[i][0]) == type(-1):
            continue
        A += a[i]
        B += b[i]
        C += c[i]
        D += d[i]
    return A, B, C, D


def unpack_reward(elements):
    a, b = elements
    return np.array(a), np.array(b)

def get_LIST(args):
    path_to_miplib = os.path.join(
        args.path_to_data_dir, 
        args.instance
    )
    tmp_list = os.listdir(path_to_miplib)
    path_to_sequence = os.path.join(
        args.path_to_data_dir, 
        args.instance,
        "sequence.npy"
    )
    sequence = np.load(path_to_sequence)
    for i in range(args.index_start, args.index_end):
        instances_list.append(tmp_list[sequence[i]])
    args.index_start = 0
    args.index_end = len(instances_list)

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Settings.
    parser.add_argument('--instances', type=str, help="the MILP class")
    parser.add_argument('--savedir', type=str, default='./model', help="the directory to save the model")
    parser.add_argument('--n_cpus', type=int, default=48, help="number of available CPUs (for parallel processing)")
    parser.add_argument('--path_to_data_dir', type=str, default="./data", help="directory of the data")
    
    # training & bandit settings
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--bandit_ins_num', type=int, default=6, help="the number of instances sampled per iteration")
    parser.add_argument('--bandit_epoch', type=int, default=100, help="the number of epochs for training the bandit")
    parser.add_argument('--train_epoch', type=int, default=100, help="the number of epochs for training the model")
    parser.add_argument('--logdir', type=str, default='./log/', help="the directory to save the log")
    parser.add_argument('--action_sampled_num', type=int, default=8, help="the number of actions sampled per instance")
    parser.add_argument('--lr_decay', type=float, default=1, help="the learning rate decay")
    parser.add_argument('--train_index_end', type=int, default=800, help="the number of instances used for training")
    parser.add_argument('--reward_avg_time', type=int, default=3, help="the number of interactions with the environment to average the reward")
    parser.add_argument('--max_train_set', type=int, default=999999999, help="the maximum number of context-action-reward pairs stored in the training set")
    parser.add_argument('--ucb_nu', type=float, default=1.5/1.6, help='nu for control variance')
    parser.add_argument('--ucb_lamb', type=float, default=0.001, help='lambda for regularzation')
    parser.add_argument('--ucb_on', type=int, default=1, help='whether to use ucb durting training; set to 1')
    parser.add_argument('--ucb_val_on', type=int, default=1, help='whether to use ucb during validation')
    parser.add_argument('--actionsdir', type=str, default='./restricted_space', help="the directory to the restricted action space")
    parser.add_argument('--actions_name', type=str, default='restricted_actions_space.npy', help="the name of the restricted action space")
    
    # NN architecture settings
    parser.add_argument('--embsize', type=int, default=64, help='the embedding size')
    
    # validation settings
    parser.add_argument('--val_index_start', type=int, default=800, help='the index of the first instance used for validation')
    parser.add_argument('--val_index_end', type=int, default=900, help='the index of the last instance used for validation')
    parser.add_argument('--val_loss_freq', type=int, default=10, help='the frequency of calculating the validation loss')
    
    # setting for k2
    parser.add_argument('--Z_0_path', type=str, help="the path to the Z matrix for the first configuration")
    parser.add_argument('--model_0_path', type=str, help="the path to the model for the first configuration")
    parser.add_argument('--step_k2', type=int, default=5, help="the value of n2")
    
    args = parser.parse_args()
    args.n_cpus = min(args.n_cpus, _cpu_count())
    print(f'{args.n_cpus} cpus available')

    if args.instances == "miplib":
        args = get_LIST(args)
    
    np.random.seed(42)
    torch.manual_seed(42)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    writer = SummaryWriter(args.logdir)
    writer_train_len = 0
    embsize = args.embsize

    path_to_actions = args.actionsdir + f"/{args.instances}/{args.actions_name}"
    action_space = np.load(path_to_actions)
    
    bandit_epoch = args.bandit_epoch
    bandit_ins_num = args.bandit_ins_num
    Bandit0 = _b_model.NeuralUCB("Neural_UCB", args.ucb_lamb, args.ucb_nu, action_space, args)
    Bandit0.model = _b_model.getModel("Neural_UCB")
    Bandit0.model.load_state_dict(torch.load(args.model_0_path))
    Bandit0.model.train(False)
    Bandit0.U = torch.load(args.Z_0_path)
    Bandits = [(Bandit0, 0)]
    Bandit = _b_model.NeuralUCB("Neural_UCB", args.ucb_lamb, args.ucb_nu, action_space, args)
    train_set = []
    for t in range(bandit_epoch):
        print("iteration: ", t)

        # log: val
        index_val = range(args.val_index_start, args.val_index_end)
        outputs = Parallel(n_jobs=args.n_cpus)(
            delayed(multiprocess_context_helper)(args_) for args_ in list(
                zip(
                    index_val,
                    [Bandits] * len(index_val),
                    [Bandit] * len(index_val),
                    [True] * len(index_val)
                )
            )
        )
        action_contexts, actions, paths, process_nums = unpack_action_contexts(zip(*outputs), len(index_val))
        outputs = Parallel(n_jobs=args.n_cpus)(
            delayed(multiprocess_solve_helper)(args_) for args_ in list(
                zip(
                    range(len(action_contexts)),
                    process_nums,
                    actions,
                    paths
                )
            )
        )
        rewards, improvs = unpack_reward(zip(*outputs))
        improvs_val = np.array(improvs)
        q3, q1 = np.percentile(improvs_val, [75, 25])
        improvs_val_IQR = improvs_val[(improvs_val > q1) * (improvs_val < q3)]
        writer.add_scalars(
            'val_mean_median_IQM',
            {
                "mean": improvs_val.mean(),
                "median": np.median(improvs_val),
                "IQM": improvs_val_IQR.mean()
            },
            t
        )
        writer.add_scalar('val_%', (improvs_val > -1e-9).sum() / improvs_val.size, t)
        writer.add_scalar('val_std', improvs_val.std(), t)

        val_set = []
        for action_context, reward in zip(action_contexts, rewards):
            val_set.append([action_context, reward])
            
        # train: parallel sample different problems
        outputs = Parallel(n_jobs=args.n_cpus)(
            delayed(multiprocess_context_helper)(args_) for args_ in list(
                zip(
                    range(bandit_ins_num),
                    [Bandits] * bandit_ins_num,
                    [Bandit] * bandit_ins_num,
                    [False] * bandit_ins_num
                )
            )
        )
        action_contexts, actions, paths, process_nums = unpack_action_contexts(zip(*outputs), bandit_ins_num)
        outputs = Parallel(n_jobs=args.n_cpus)(
            delayed(multiprocess_solve_helper)(args_) for args_ in list(
                zip(
                    range(len(action_contexts)),
                    process_nums,
                    actions,
                    paths
                )
            )
        )
        rewards, improvs = unpack_reward(zip(*outputs))
        for index in range(len(action_contexts)):
            train_set.append([action_contexts[index], rewards[index]])

        print("training the model...")
        Bandit.model = train(train_set, val_set, Bandit.model, args)
        
        # save the model
        os.makedirs(args.savedir + f"/{args.instances}_k2", exist_ok=True)
        model_path = args.savedir + f"/{args.instances}_k2/model-{t}"
        Z_path = args.savedir + f"/{args.instances}_k2/Z-{t}"
        saveModel(Bandit.model, model_path)
        torch.save(Bandit.U, Z_path)
