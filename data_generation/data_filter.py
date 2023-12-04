import os
import pyscipopt as pyopt
import argparse
from joblib import Parallel, delayed
from multiprocessing import cpu_count as _cpu_count
import numpy as np
import shutil

path_to_randomness_control_set = '../SCIP_settings/randomness_control.set'
instances_list = []

def get_LIST(indexes):
    global instances_list
    path_to_miplib = os.path.join(
        args.path_to_raw_data_dir, 
        args.instances
    )
    tmp_list = os.listdir(path_to_miplib)
    for i in indexes:
        instances_list.append(tmp_list[i])

def multiprocess_helper(helper_args):
    index, args = helper_args
    print(f"{index}th {args.instances} instance starts!")
    instance_suffix = f"model-{index}/model.mps"
    if args.instances == "nnv":
        instance_suffix = f"model-{index}/model.proto.lp"
    elif args.instances == "load_balancing":
        instance_suffix = f"load_balancing_{index}.mps.gz"
    elif args.instances == "miplib":
        instance_suffix = instances_list[index]
        
    path_to_problem = os.path.join(
        args.path_to_raw_data_dir, 
        args.instances, 
        instance_suffix
    )
    
    model = pyopt.Model()
    model.hideOutput(1)
    model.readProblem(path_to_problem)
    model.readParams(path_to_randomness_control_set)
    model.setParam("limits/time", args.time_limit)
    model.optimize()
    
    if model.getGap() > args.gap_limit+1e-6:
        print(f"{index}th instance is flitered out!")
    else:
        dst_path = os.path.join(
            args.path_to_data_dir, 
            args.instances, 
            instance_suffix
        )
        os.makedirs(dst_path, exist_ok=True)
        shutil.copy(path_to_problem, dst_path)
        print(f"{index}th instance is selected!")
    return model.getGap()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', type=str, help="MILP class to be filtered")
    parser.add_argument('--n_cpus', type=int, default=48, help="number of available CPUs (for parallel processing)")
    parser.add_argument('--path_to_raw_data_dir', type=str, default="../data/raw", help="directory of the downloaded raw data")
    parser.add_argument('--path_to_data_dir', type=str, default="../data", help="save directory of the filtered data")
    parser.add_argument('--time_limit', type=float, default=float('inf'), help="time limit for solving each instance")
    parser.add_argument('--gap_limit', type=float, default=0.0, help="gap limit for filtering instances")
    parser.add_argument('--index_start', type=int, default=0, help="index of the first instance to consider")
    parser.add_argument('--index_end', type=int, default=1000, help="total number of the instances to consider")
    args = parser.parse_args()
    
    indexes = range(args.index_start, args.index_end)
    
    if args.instances == "miplib":
        get_LIST(indexes)
    
    outputs = Parallel(n_jobs=args.n_cpus)(
        delayed(multiprocess_helper)(args_) for args_ in list( 
            zip(
                indexes,
                [args] * len(indexes)
            )
        )
    )