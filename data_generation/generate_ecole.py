from collections import OrderedDict
import ecole
import os
import numpy as np
import json
import argparse
# import pyscipopt as pyopt

INSTANCES = [
        ('combauct', OrderedDict({'n_items': 100, 'n_bids': 500}), 'max'),
        ('capfac', OrderedDict({'n_customers': 100, 'n_facilities': 100}), 'min'),
        ('indset', OrderedDict({'n_nodes': 500}), 'max'),
    ]
MAXNUM = 1000

def main(args):
    for ptype, kwargs, sense in INSTANCES:
        spec = '-'.join([str(i) for i in kwargs.values()])
        path = f'{args.path_to_data_dir}/{ptype}-{spec}'

        for i in range(MAXNUM):
            generator, new_kwargs = get_generator(ptype, **kwargs)
            modeldir = os.path.join(path, f'model-{i}')
            os.makedirs(modeldir, exist_ok=True)
            json_object = json.dumps(new_kwargs)
            #Writing to sample.json
            with open(os.path.join(modeldir, 'info.json'), "w") as outfile:
                outfile.write(json_object)
            model = next(generator) # .as_pyscipopt()
            if sense == 'max':  # Revert sense, so we always have minimization problems for code.
               new_obj = -1 * model.getObjective()
               model.setObjective(new_obj, 'minimize')
            model.write_problem(os.path.join(modeldir, 'model.mps'))

        print(f'Generated {MAXNUM} instances of {ptype}-{spec}.')


def get_generator(ptype, **kwargs):

    GENERATORS = {
        'setcover': ecole.instance.SetCoverGenerator,
        'combauct': ecole.instance.CombinatorialAuctionGenerator,
        'capfac': ecole.instance.CapacitatedFacilityLocationGenerator,
        'indset': ecole.instance.IndependentSetGenerator,
    }

    ecole.seed(0)
    if ptype == "setcover":
        kwargs['density'] = 0.05 + np.random.rand() * 0.1
        kwargs['max_coef'] = 100 + np.random.randint(0,200)
    elif ptype == "indset":
        type = "barabasi_albert" if np.random.randint(0,2) == 0 else "erdos_renyi"
        if type == "barabasi_albert":
            kwargs['affinity'] = np.random.randint(2,7)
        else:
            kwargs['graph_type'] = type
            kwargs['edge_probability'] = np.random.rand() * 0.005 + 0.005
    elif ptype == "combauct":
        kwargs['value_deviation'] = 0.25 + np.random.rand() * 0.5
        kwargs['add_item_prob'] = 0.5 + np.random.rand() * 0.25
        kwargs['max_n_sub_bids'] = np.random.randint(3,8)
        kwargs['additivity'] = -0.1 + np.random.rand() * 0.5
        kwargs['budget_factor'] = 1.25 + np.random.rand() * 0.5
        kwargs['resale_factor'] = 0.35 + np.random.rand() * 0.3
    generator =  GENERATORS[ptype](**kwargs)
    generator.seed(0)

    return generator, kwargs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data_dir', type=str,
                        default='../data',)
    args = parser.parse_args()
    main(args)