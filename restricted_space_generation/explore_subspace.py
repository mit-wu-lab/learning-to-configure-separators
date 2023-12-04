import argparse
import os

import random

from joblib import Parallel, delayed
from multiprocessing import cpu_count as _cpu_count
import pyscipopt as pyopt
import numpy as np
import copy
import math

path_to_randomness_control_set = '../SCIP_settings/randomness_control.set'
error_signal = -999999999
instances_list = []

def solve(path_to_problem, action=None,default=-1):
    sepa_list = [
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
    
    # get the solve time of SCIP
    model = pyopt.Model()
    model.hideOutput(1)
    model.readProblem(path_to_problem)
    model.readParams(path_to_randomness_control_set)
    model.setParam("limits/gap", args.gap_limit)

    if default == -1:
        model.optimize()
        SCIP_time = model.getSolvingTime()
        return SCIP_time

    for index in range(len(sepa_list)):
        on_or_off = 1
        # data collection: search trajectory
        if action[index] < 0.5:
            on_or_off = -1
        model.setParam(f'separating/{sepa_list[index]}/freq', on_or_off)
    model.setParam("limits/time", default*2.5)
    model.optimize()
    time = model.getSolvingTime()

    improv = (default - time)/default
    return improv, time

def multiprocess_helper(helper_args):
    index, action, action_index = helper_args
    print(f"process {index}: start!")
    
    global args
    instance_suffix = f"model-{index}/model.mps"
    if args.instances == "nnv":
        instance_suffix = f"model-{index}/model.proto.lp"
    elif args.instances == "load_balancing":
        instance_suffix = f"load_balancing_{index}.mps.gz"
    elif args.instances == "miplib":
        instance_suffix = instances_list[index]
        
    path_to_problem = os.path.join(
        args.path_to_data_dir, 
        args.instances, 
        instance_suffix
    )
    
    if os.path.exists(path_to_problem) == False:
        print(path_to_problem)
        return error_signal, error_signal, index, action_index
    
    default_time = 0
    repeat_time = 3
    for i in range(repeat_time):
        default_time += solve(
            path_to_problem,
            action=None,
            default=-1
        ) / repeat_time
    print(f"process {index}, {action_index}: finish default")
        
    action_improv = 0
    action_time = 0
    for i in range(repeat_time):
        action_improv_cur, action_time_cur = solve(
            path_to_problem,
            action=action,
            default=default_time
        ) 
        action_improv += action_improv_cur / repeat_time
        action_time += action_time_cur / repeat_time
    print(f"process {index}, {action_index}: finish emprical mask")
    return action_improv, action_time, index, action_index

def find_actions_near_zero(factor, index, action, actions):
    if index >= 17 or action.sum() >= factor:
        actions.append(copy.deepcopy(action))
        return actions

    action[index] = 0
    actions = find_actions_near_zero(factor, index+1, copy.deepcopy(action), copy.deepcopy(actions))
    action[index] = 1
    actions = find_actions_near_zero(factor, index+1, copy.deepcopy(action), copy.deepcopy(actions))
    
    return actions

def getMasks(args):
    actions = []
    path_to_constant = args.path_to_ins_agnst + '/' + args.instances + "/best_ins_agnostic.npy"
    best_constant = np.load(path_to_constant)
    if args.mode == "mask_space":
        one_index = []
        for i in range(17):
            if best_constant[i] > 0.5:
                one_index.append(i)
        for i in range(2**len(one_index)):
            action = np.zeros((17,))
            now = i
            for j in one_index:
                action[j] = now % 2
                now = int(now/2)
            actions.append(action)
    elif args.mode == "near_zero" or args.mode == "near_mask":
        actions_helper = find_actions_near_zero(args.near_factor, 0, np.zeros(17), [])
        if args.mode == "near_mask":
            actions = np.array([best_constant] * len(actions_helper))
            for i, indicator in enumerate(actions_helper):
                for j in range(17):
                    if indicator[j] > 0.5:    
                        actions[i][j] = 1 - actions[i][j]
        else:
            actions = actions_helper
        
    print("finish finding the actions, the number of actions is: ", len(actions))
    return actions

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
    parser.add_argument('--mode', type=str, help="the sample strategy to apply")
    parser.add_argument('--n_cpus', type=int, default=48, help="number of available CPUs (for parallel processing)")
    parser.add_argument('--path_to_data_dir', type=str, default="../data", help="directory of the data")
    parser.add_argument('--path_to_ins_agnst', type=str, default="../restricted_space/", help="directory of the best instance-agnostic configuration")
    parser.add_argument('--save_dir', type=str, default="../restricted_space/", help="save directory of sampled configurations")
    parser.add_argument('--gap_limit', type=float, default=0.0, help="gap limit for solving instances")
    parser.add_argument('--near_factor', type=int, default=3, help="this parameter controls how large we explore the space")
    parser.add_argument('--index_start', type=int, default=0, help="index of the first instance to consider")
    parser.add_argument('--index_end', type=int, default=100, help="total number of the instances to consider")
    args = parser.parse_args()
    args.n_cpus = min(args.n_cpus, _cpu_count())

    if args.instances == "miplib":
        args = get_LIST(args)
        args.index_end = 30
    
    index_start = args.index_start
    index_end = args.index_end
    
    # For the masks
    actions = getMasks(args)
    a_len = len(actions)
    t_len = (index_end - index_start) * a_len
    best_constants = []
    indexes = []
    masks_index = []
    for i in range(a_len):
        for j in range(index_start, index_end):
            best_constants.append(actions[i])
            indexes.append(j)
            masks_index.append(i)
            
    improvs = None    
    outputs = Parallel(n_jobs=args.n_cpus)(
        delayed(multiprocess_helper)(args_) for args_ in list( 
            zip(
                indexes,
                best_constants,
                masks_index
            )
        )
    )
    
    improvs, time_abs, return_index, return_mask_index = zip(*outputs)
    stats = np.zeros((a_len, index_end-index_start))
    for i in range(t_len):
        stats[return_mask_index[i], return_index[i] - index_start] = improvs[i]
    for i in range(index_start, index_end):
        if stats[:,index_end-i-1].sum() <= error_signal+1:
            stats = np.delete(stats, index_end-i-1, axis=1)
    path_to_save_dir = args.save_dir + "/" + args.instances + "/" + args.mode
    os.makedirs(path_to_save_dir, exist_ok=True)
    with open(path_to_save_dir + "/stats.npy", "wb") as f:
        np.save(f, stats)
    with open(path_to_save_dir + "/configs.npy", "wb") as ff:
        np.save(ff, np.array(actions))