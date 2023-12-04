import json
import numpy as np
import os
import pickle
import joblib
import platform
import torch

import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

def multiprocess(func, tasks, cpus=None):
    if cpus == 1 or len(tasks) == 1:
        return [func(t) for t in tasks]
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.imap(func, tasks))

def multithread(func, tasks, cpus=None, show_bar=True):
    bar = lambda x: tqdm(x, total=len(tasks)) if show_bar else x
    if cpus == 1 or len(tasks) == 1:
        return [func(t) for t in bar(tasks)]
    with ThreadPool(cpus or os.cpu_count()) as pool:
        return list(bar(pool.imap(func, tasks)))

SEPA2IX = {
  'aggregation': 0,
  'clique': 1,
  'disjunctive': 2,
  'gomory': 3,
  'impliedbounds': 4,
  'mcf': 5,
  'oddcycle': 6,
  'strongcg': 7,
  'zerohalf': 8,
}

IX2SEPA = {ix: sepa for sepa, ix in SEPA2IX.items()}

SAVE_DIR = '../exp_optlearn/runs/exp_optdata' # TODO: need to double check

def get_data_directory(args): # TODO: need to double check
    instances = os.path.basename(os.path.normpath(os.path.dirname(args.path_to_instance)))
    model = os.path.basename(os.path.normpath(args.path_to_instance))
    data_directory = os.path.join(
        SAVE_DIR,
        instances,
        args.config,
        model,
        args.state,
        args.samplingstrategy,
        f'offset-{args.random_offset}',
    )
    os.makedirs(data_directory, exist_ok=True)
    return data_directory

def save_state(path, state):
    os.makedirs(path, exist_ok=True)

    for k, obj in state.items():
        obj_path = os.path.join(path, k)

        if isinstance(obj, np.ndarray):
            save_func = save_numpy
            obj_path += '.npy'
        else:
            save_func = save_pickle
            obj_path += '.pkl'

        save_func(obj_path, obj)


def save_json(path, d):
    with open(path, 'w') as file:
        json.dump(d, file, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        d = json.load(f)
    return d

def save_numpy(path, arr):
    with open(path, 'wb') as file:
        np.save(file, arr)

def load_numpy(path):
    with open(path, 'rb') as file:
        arr = np.load(path)
    return arr

def save_pickle(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file, protocol=5)

def load_pickle(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj

def save_joblib(path, obj):
    joblib.dump(obj, path)

def load_joblib(path):
    return joblib.load(path)


class ExitStatus:
    CONVERGED_DUAL = 0
    CONVERGED_INTEGRAL = 1
    MAXITER = 2
    ERROR_TRIVIAL = 3
    ERROR_SEPARATION = 4
    ERROR_NOCUTS = 5
    ERROR_DUAL = 6
    ERROR_SIGN_SWITCH = 7
    ERROR_IGC_MESS = 8
    ERROR_DEF_JSON_MISSING = 9
    ERROR_EXP_JSON_MISSING = 10


#############################################################
# exp_optlearn util
#############################################################
def save_json(path, d):
    with open(path, 'w') as file:
        json.dump(d, file, indent=4)


def load_json(path):
    with open(path, 'r') as f:
        d = json.load(f)
    return d


def save_pickle(path, d):
    with open(path, 'wb') as file:
        pickle.dump(d, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    return d

def save_numpy(path, arr):
    with open(path, 'wb') as file:
        np.save(file, arr)

def load_numpy(path):
    with open(path, 'rb') as file:
        arr = np.load(path)
    return arr

def save_joblib(path, obj):
    joblib.dump(obj, path)

def load_joblib(path):
    return joblib.load(path)

def find_all_paths_sepa(
        split,
        instances,
        state,
        data_dir='runs'
        ):

    paths = []

    basepath = f'./{data_dir}/{instances}_{split}'

    models = []
    if not os.path.exists(basepath): return paths
    for f in os.listdir(basepath):
        if f.startswith('model-'):
            models.append(f)

    for model in models:
        model_path = f'{basepath}/{model}/{state}/'
        for root, dirs, _ in os.walk(model_path):
            for d in dirs:
                if d.startswith('nsepacut'):
                    path = f'{root}/{d}'
                    paths.append(path)

    return paths

def find_all_paths(
        split,
        instances,
        config,
        state,
        samplingstrategy,
        ):

    paths = []

    basepath = f'./runs/exp_optdata/{instances}_{split}'
    if is_local():
        basepath += 'mock'
    basepath = f'{basepath}/{config}'

    models = []
    if not os.path.exists(basepath): return paths
    for f in os.listdir(basepath):
        if f.startswith('model-'):
            models.append(f)

    for model in models:
        model_path = f'{basepath}/{model}/{state}/{samplingstrategy}'
        for root, dirs, _ in os.walk(model_path):
            for d in dirs:
                if d.startswith('nsepacut'):
                    path = f'{root}/{d}'
                    paths.append(path)

    return paths

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def is_local():
    return platform.system() == 'Darwin'

"""
Utility functions to load and save torch model checkpoints 
"""
def load_checkpoint(net, optimizer=None, step='max', save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    checkpoints = [x for x in os.listdir(save_dir) if not x.startswith('events')]
    if step == 'max':
        step = 0
        if checkpoints:
            step, last_checkpoint = max([(int(x.split('.')[0]), x) for x in checkpoints])
    else:
        last_checkpoint = str(step) + '.pth'
    if step:
        save_path = os.path.join(save_dir, last_checkpoint)
        state = torch.load(save_path, map_location='cpu')
        net.load_state_dict(state['net'])
        if optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Loaded checkpoint %s' % save_path)
    return step

def save_checkpoint(net, optimizer, step, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, str(step) + '.pth')

    torch.save(dict(net=net.state_dict(), optimizer=optimizer.state_dict()), save_path)
    print('Saved checkpoint %s' % save_path)


if __name__ == '__main__':

    paths = find_all_paths(
        'train',
        'binpacking-66-66',
        'all-def-def',
        'learn1',
        'mixed1'
    )
    print(len(paths))
    import pdb; pdb.set_trace()
