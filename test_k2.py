import argparse
import os

import random

import utils as _utils
import torch
import torch.nn as nn

from joblib import Parallel, delayed
from multiprocessing import cpu_count as _cpu_count
import pyscipopt as pyopt
import numpy as np
import copy
import math
import time

import context as _context

import model as _b_model

path_to_randomness_control_set = './SCIP_settings/randomness_control.set'
error_signal = -999999999
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
        execution_time_tmp = time.time()
        if self.sepa_round in self.actions.keys():
            action = self.actions[self.sepa_round]
            for index in range(len(self.sepa_list)):
                on_or_off = 1
                if action[index, 0] == 0:
                    on_or_off = -1
                self.model.setParam(f'separating/{self.sepa_list[index]}/freq', on_or_off)
        self.sepa_round += 1
        self.execution_time +=  time.time() - execution_time_tmp
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
        self.execution_time = 0
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

def solve(path_to_problem, actions=None,default=-1):     
    # get the solve time of SCIP
    model = pyopt.Model()
    model.hideOutput(1)
    model.readProblem(path_to_problem)
    model.readParams(path_to_randomness_control_set)
    if default == -1:
        sepa_manager_SM = SepaManager_SM()
        sepa_manager_SM.addModel(model, actions={})
        model.optimize()
        SCIP_time = model.getSolvingTime() - sepa_manager_SM.execution_time
        return SCIP_time

    sepa_manager_SM = SepaManager_SM()
    sepa_manager_SM.addModel(model, actions=actions)
    model.setParam(f'limits/time', default * 4)
    model.optimize()
    time = model.getSolvingTime() - sepa_manager_SM.execution_time

    improv = (default - time)/default
    return improv, time

def multiprocess_helper(helper_args):
    index, Bandit, Bandits = helper_args
    print(f"process {index}: start!")
    
    repeat_time = 3
    instance_suffix = f"model-{index}/model.mps"
    if args.instances == "nnv":
        instance_suffix = f"model-{index}/model.proto.lp"
    elif args.instances == "load_balancing":
        instance_suffix = f"load_balancing_{index}.mps.gz"
    elif args.instances == "miplib":
        instance_suffix = instances_list[index]
        
    path_to_problem = os.path.join(
        args.path_to_data_dir, 
        args.instances+'_test', 
        instance_suffix
    )
    
    if os.path.exists(path_to_problem) == False:
        return error_signal, error_signal
        
    actions = None
    input_context, actions = _context.getInputContextAndInstruction(path_to_problem, args.step_k2, args.instances, Bandits=Bandits)
    if input_context == None:
        return error_signal, error_signal
    best_action, _ = Bandit.getActions(input_context, num=1, eva=True)
    best_action = best_action[0]
    actions[args.step_k2] = best_action
    
    print(f"For model-{index}, the best action:")
    print(f"step 0: {actions[0][:,0]}") 
    print(f"step {args.step_k2}: {actions[args.step_k2][:,0]}") 
    
    default_time = 0
    for i in range(repeat_time):
        default_time += solve(
            path_to_problem,
            actions=None,
            default=-1
        ) / repeat_time
    print(f"process {index}: finish default")
    
    our_improv = 0
    our_time = 0
    for i in range(repeat_time):
        our_improv_cur, our_time_cur = solve(
            path_to_problem,
            actions=actions,
            default=default_time
        ) 
        our_improv += our_improv_cur / repeat_time
        our_time += our_time_cur / repeat_time
    print(f"process {index}: finish emprical mask")
    return our_improv, our_time

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
    parser.add_argument('--n_cpus', type=int, default=48, help="number of available CPUs (for parallel processing)")
    parser.add_argument('--path_to_data_dir', type=str, default="./data", help="directory of the data")
    
    parser.add_argument('--actionsdir', type=str, default='./restricted_space', help="the directory to the restricted action space")
    parser.add_argument('--actions_name', type=str, default='restricted_actions_space.npy', help="the name of the restricted action space")
    parser.add_argument('--index_start', type=int, default=900, help="the start index of the test instances")
    parser.add_argument('--index_end', type=int, default=1000, help="the end index of the test instances")
    parser.add_argument('--ucb_nu', type=float, default=1.5/1.6, help='nu for control variance')
    parser.add_argument('--ucb_lamb', type=float, default=0.001, help='lambda for regularzation')
    parser.add_argument('--ucb_on', type=int, default=1, help='whether to use ucb during training; set to 1')
    parser.add_argument('--ucb_val_on', type=int, default=1, help='whether to use ucb during validation')
    
    # k = 2
    parser.add_argument('--Z_1_path', type=str, help="the path to the Z matrix at step n2")
    parser.add_argument('--model_1_path', type=str, help="the path to the model at step n2")
    parser.add_argument('--Z_0_path', type=str, help="the path to the Z matrix at step n1")
    parser.add_argument('--model_0_path', type=str, help="the path to the model at step n1")
    parser.add_argument('--step_k2', type=int, default=5, help="the value of n2")
    
    args = parser.parse_args()
    args.n_cpus = min(args.n_cpus, _cpu_count())
    
    if args.instances == "miplib":
        args = get_LIST(args)

    np.random.seed(42)
    torch.manual_seed(42)
    action_space = None
    nn_model = None
    Bandit = None
    Bandits = None

    path_to_actions = args.actionsdir + f"/{args.instances}/{args.actions_name}"
    action_space = np.load(path_to_actions)
    nn_model = _b_model.getModel("Neural_UCB")
    nn_model.load_state_dict(torch.load(args.model_1_path))
    nn_model.train(False)
    U = torch.load(args.Z_1_path)
    Bandit = _b_model.NeuralUCB("Neural_UCB", args.ucb_lamb, args.ucb_nu, action_space, args)
    Bandit.U = U
    Bandit.model = nn_model
    
    Bandit0 = _b_model.NeuralUCB("Neural_UCB", args.ucb_lamb, args.ucb_nu, action_space, args)
    Bandit0.model = _b_model.getModel("Neural_UCB")
    Bandit0.model.load_state_dict(torch.load(args.model_0_path))
    Bandit0.model.train(False)
    Bandit0.U = torch.load(args.Z_0_path)
    Bandits = [(Bandit0, 0)]
    
    index_start = args.index_start
    index_end = args.index_end   
    outputs = Parallel(n_jobs=args.n_cpus)(
        delayed(multiprocess_helper)(args_) for args_ in list( 
            zip(
                range(index_start, index_end),
                [Bandit] * (index_end - index_start),
                [Bandits] * (index_end - index_start),
            )
        )
    )
    
    improvs, time_abs = zip(*outputs)
    improvs = np.array(improvs)
    time_abs = np.array(time_abs)
    improvs = improvs[time_abs>-1e-6]
    time_abs = time_abs[time_abs>-1e-6]
    
    name = "ours k=2"
    data = improvs
    data = np.array(data)
    q3, q1 = np.percentile(data, [75 ,25])
    data_IQR = data[(data > q1)*(data < q3)]
    print(f"---------------------------------")
    print(f"For {name}:")
    print(f"The mean: {np.around(data.mean(),3)}")
    print(f"The median: {np.around(np.median(data),3)}")
    print(f"The IQM: {np.around(data_IQR.mean(),3)}")
    print(f"The % of improvement: {np.around((data > -1e-9).sum()/data.size,3)}")
    print(f"The std: {np.around(data.std(),3)}")
    # print(data)
    
    q3_abs, q1_abs = np.percentile(time_abs, [75 ,25])
    time_abs_IQR = time_abs[(time_abs > q1_abs)*(time_abs < q3_abs)]
    print(f"The abs mean: {np.around(time_abs.mean(),3)}")
    print(f"The abs median: {np.around(np.median(time_abs),3)}")
    print(f"The abs IQM: {np.around(time_abs_IQR.mean(),3)}")
    print(f"The abs std: {np.around(time_abs.std(),3)}")
    