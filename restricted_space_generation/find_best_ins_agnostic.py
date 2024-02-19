import numpy as np
import torch
import torch.nn as nn
import pyscipopt as pyopt
import argparse
import os
from joblib import Parallel, delayed
from multiprocessing import cpu_count as _cpu_count

path_to_randomness_control_set = '../SCIP_settings/randomness_control.set'
instances_list = []

def solve(path_to_problem, action=None, default=-1):
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
    return improv

def multiprocess_helper(helper_args):
    index, action = helper_args
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
    
    # for default
    default_time = 0
    repeat_time = 3
    for i in range(repeat_time):
        default_time += solve(path_to_problem,action=None,default=-1) / repeat_time
    
    # for current configuration
    improv = 0
    for i in range(repeat_time):
        improv += solve(
            path_to_problem,
            action=action,
            default=default_time
        ) / repeat_time
    
    return improv, default_time

def get_LIST(args):
    path_to_miplib = os.path.join(
        args.path_to_data_dir, 
        args.instances
    )
    tmp_list = os.listdir(path_to_miplib)
    path_to_sequence = os.path.join(
        args.path_to_data_dir, 
        args.instances,
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
    parser.add_argument('--path_to_data_dir', type=str, default="../data/", help="directory of the data")
    parser.add_argument('--configs_num', type=int, default=1000, help="number of configurations to sample")
    parser.add_argument('--index_start', type=int, default=0, help="index of the first instance to consider")
    parser.add_argument('--index_end', type=int, default=100, help="total number of the instances to consider")
    parser.add_argument('--gap_limit', type=float, default=0.0, help="gap limit for solving instances")
    parser.add_argument('--savedir', type=str, default="../restricted_space/", help="save directory of best instance-agonistic configuration")
    args = parser.parse_args()
    
    if args.instances == "miplib":
        args = get_LIST(args)
        args.index_end = 30

    best_constant_action = None
    best_score = -float("inf")
    path = args.savedir + "/" + args.instances
    os.makedirs(path, exist_ok=True)
    path_to_action = path + "/best_ins_agnostic.npy"
    path_to_score = path + "/best_improv.npy"
    for _ in range(args.configs_num):
        constant_action = np.random.randint(2,size=17)
        outputs = Parallel(n_jobs=args.n_cpus)(
            delayed(multiprocess_helper)(args_) for args_ in list( 
                zip(
                    range(args.index_start, args.index_end),
                    [constant_action] * (args.index_end - args.index_start)
                )
            )
        )
        raw_data, _ = zip(*outputs)
        score = np.mean(np.array(raw_data))
        print("----------------------")
        print(f"Current configuration is")
        print(constant_action)
        print(f"Current improv is {score}")
        if score > best_score:
            best_constant_action = constant_action
            best_score = score
            np.save(path_to_action, best_constant_action)
            np.save(path_to_score, np.array(best_score))

    print(f"Best instance agnostic is {best_constant_action}.")
    print(f"Best improvement is {best_score}.")
