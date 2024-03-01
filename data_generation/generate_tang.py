# """
# Instance generation of Tang et al. 2020
# https://arxiv.org/pdf/1906.04859.pdf
# """
#
from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import combinations
from numpy.random import default_rng
import os
import pyscipopt as pyopt
import argparse

INSTANCES = [
        ('maxcut', OrderedDict({'nV': 54, 'nE': 134}), 'max'),
        ('packing', OrderedDict({'n': 60, 'm': 60}), 'max'),
        ('binpacking', OrderedDict({'n': 66, 'm': 132}), 'max'),
]

MAXNUM = 1000

def main():

    for ptype, kwargs, sense in INSTANCES:
        generator = get_generator(ptype, **kwargs)
        spec = '-'.join([str(i) for i in kwargs.values()])
        path = f'{args.path_to_data_dir}/{ptype}-{spec}'

        for i, instance in enumerate(generator):
            model = instance  # .as_pyscipopt() not needed

            if sense == 'max':  # Revert sense, so we always have minimization problems for code.
                new_obj = -1 * model.getObjective()
                model.setObjective(new_obj, 'minimize')

            modeldir = os.path.join(path, f'model-{i}')
            os.makedirs(modeldir, exist_ok=True)
            model.writeProblem(os.path.join(modeldir, 'model.mps'))

            if (i + 1) >= MAXNUM:
                break

        print(f'Generated {MAXNUM} instances of {ptype}-{spec}.')


def get_generator(ptype, **kwargs):

    GENERATORS = {
        'maxcut': MaxCutGenerator,
        'packing': PackingGenerator,
        'binpacking': BinPackingGenerator,
        'planning': PlanningGenerator,
        'maxcutplus': MaxCutGeneratorPlus
    }

    generator = GENERATORS[ptype](**kwargs, seed=42)

    return generator


class Generator(ABC):

    def __init__(self, seed=42):
        self.num = 0
        self.rng = default_rng(seed=seed)

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass

    @staticmethod
    @abstractmethod
    def generate_instance(self, *args, **kwargs):
        pass


# Max Cut
# The size (n,m) of the final relaxation (excluding non-negativity constraints)
# is determined by the underlying graph G=(V,E), as n = |V|+|E| and m = 3|E|+|V|.
# Medium size instances have |V|=7 and |E|=20 (note: it basically keeps all but one edge?).

class MaxCutGenerator(Generator):

    def __init__(self, nV, nE, seed=42):
        super().__init__(seed=seed)
        self.nV = nV
        self.nE = nE

    def __next__(self):
        instance = self.generate_instance(nV=self.nV, nE=self.nE, num=self.num, rng=self.rng)
        self.num += 1
        return instance

    @staticmethod
    def generate_instance(nV, nE, num, rng):

        name = 'model-'+str(num)
        # sample nE edges from all the possible ones
        V = list(range(1, nV+1))
        allE = list(combinations(V, 2))
        E = list(map(tuple, rng.choice(allE, nE, replace=False)))
        E.sort()

        # sample weights for E
        W = dict.fromkeys(E)
        for e in W.keys():
            W[e] = rng.integers(low=0, high=10, endpoint=True)  # as in Tang et al.

        # build the optimization model
        model = pyopt.Model(name)

        x, y = {}, {}
        for u in V:
            x[u] = model.addVar(vtype='B', lb=0.0, ub=1, name="x(%s)" % u)
        for e in E:  # e=(u,v)
            y[e] = model.addVar(vtype='B', lb=0.0, ub=1, name="y(%s,%s)" % (e[0], e[1]))
            model.addCons(y[e] <= x[e[0]] + x[e[1]], "C1(%s,%s)" % (e[0], e[1]))
            model.addCons(y[e] <= 2 - x[e[0]] - x[e[1]], "C2(%s,%s)" % (e[0], e[1]))

        # objective is max(c^T x)
        model.setObjective(pyopt.quicksum(W[e]*y[e] for e in E), "maximize")

        return model

class MaxCutGeneratorPlus(Generator):

    def __init__(self, nV, nE, seed=42):
        super().__init__(seed=seed)
        self.nV = nV
        self.nE = nE

    def __next__(self):
        instance = self.generate_instance(nV=self.nV, nE=self.nE, num=self.num, rng=self.rng)
        self.num += 1
        return instance

    @staticmethod
    def generate_instance(nV, nE, num, rng):

        name = 'model-'+str(num)
        # sample nE edges from all the possible ones
        V = list(range(1, nV+1))
        allE = list(combinations(V, 2))
        E = list(map(tuple, rng.choice(allE, nE, replace=False)))
        E.sort()

        # sample weights for E
        W = dict.fromkeys(E)
        for e in W.keys():
            W[e] = rng.integers(low=0, high=100, endpoint=True)  # as in Tang et al.

        # build the optimization model
        model = pyopt.Model(name)

        x, y = {}, {}
        for u in V:
            x[u] = model.addVar(vtype='B', lb=0.0, ub=1, name="x(%s)" % u)
        for e in E:  # e=(u,v)
            y[e] = model.addVar(vtype='B', lb=0.0, ub=1, name="y(%s,%s)" % (e[0], e[1]))
            model.addCons(y[e] <= x[e[0]] + x[e[1]], "C1(%s,%s)" % (e[0], e[1]))
            model.addCons(y[e] <= 2 - x[e[0]] - x[e[1]], "C2(%s,%s)" % (e[0], e[1]))

        # objective is max(c^T x)
        model.setObjective(pyopt.quicksum(W[e]*y[e] for e in E), "maximize")

        return model

# Packing
# Packing problem with non-negative general integer variables.
# Dimension is directly determined by n=number of items and m=number of resource constraints.

class PackingGenerator(Generator):

    def __init__(self, n, m, seed=42):
        super().__init__(seed=seed)
        self.n = n
        self.m = m

    def __next__(self):
        instance = self.generate_instance(n=self.n, m=self.m, num=self.num, rng=self.rng)
        self.num += 1
        return instance

    @staticmethod
    def generate_instance(n, m, num, rng):

        name = 'model-'+str(num)

        # sample coefficients, ranges as in Tang et al.
        A = rng.integers(low=0, high=5, size=(m, n), endpoint=True)
        b = rng.integers(low=9*n, high=10*n, size=m, endpoint=True)
        c = rng.integers(low=1, high=10, size=n, endpoint=True)

        # build the optimization model
        model = pyopt.Model(name)

        x = {}
        for j in range(n):
            x[j] = model.addVar(vtype='I', lb=0.0, ub=None, name="x(%s)" % (j+1))
        for i in range(m):
            model.addCons(pyopt.quicksum(A[i][j]*x[j] for j in range(n)) <= b[i], "Resource(%s)" % (i+1))

        # objective is max(c^T x)
        model.setObjective(pyopt.quicksum(c[j]*x[j] for j in range(n)), "maximize")

        return model


# Bin Packing
# Packing problem with binary variables.
# Dimension is determined by n=number of items and m=number of resource constraints.
# With respect to packing method, bounds are added for binary variables, and ranges for coefficients changed.

class BinPackingGenerator(Generator):

    def __init__(self, n, m, seed=42):
        super().__init__(seed=seed)
        self.m = m
        self.n = n

    def __next__(self):
        instance = self.generate_instance(n=self.n, m=self.m, num=self.num, rng=self.rng)
        self.num += 1
        return instance

    @staticmethod
    def generate_instance(n, m, num, rng):

        name = 'model-'+str(num)

        # sample coefficients, ranges as in Tang et al.
        A = rng.integers(low=5, high=30, size=(m, n), endpoint=True)
        b = rng.integers(low=10*n, high=20*n, size=m, endpoint=True)
        c = rng.integers(low=1, high=10, size=n, endpoint=True)

        # build the optimization model
        model = pyopt.Model(name)

        x = {}
        for j in range(n):
            x[j] = model.addVar(vtype='B', lb=0.0, ub=1, name="x(%s)" % (j+1))
        for i in range(m):
            model.addCons(pyopt.quicksum(A[i][j]*x[j] for j in range(n)) <= b[i], "Resource(%s)" % (i+1))

        # objective is max(c^T x)
        model.setObjective(pyopt.quicksum(c[j]*x[j] for j in range(n)), "maximize")

        return model


# Production Planning
# Production planning problem over planning horizon T.
# Dimension is determined as n=3T+1 and m=4T+1 (not sure how constraints are counted, maybe as 4(T+1)?)

class PlanningGenerator(Generator):

    def __init__(self, T, seed=42):
        super().__init__(seed=seed)
        self.T = T

    def __next__(self):
        instance = self.generate_instance(T=self.T, num=self.num, rng=self.rng)
        self.num += 1
        return instance

    @staticmethod
    def generate_instance(T, num, rng):

        name = 'model-'+str(num)

        # constant parameters as in Tang et al.
        # s0, sT = 0, 20  are enforced via bounds
        M = 100

        # sample coefficients, ranges as in Tang et al.
        p = rng.integers(low=1, high=10, size=T, endpoint=True)
        h = rng.integers(low=1, high=10, size=T+1, endpoint=True)
        q = rng.integers(low=1, high=10, size=T, endpoint=True)  # indices should run 1...T in objective
        # demands are not specified in the paper, so we get them as other parameters
        dval = rng.integers(low=1, high=10, size=T, endpoint=True)
        d = {}  # this makes indexing easier for constraints
        for i in range(1, T+1):
            d[i] = dval[i-1]

        # build the optimization model
        model = pyopt.Model(name)

        x, y, s = {}, {}, {}
        for i in range(T+1):
            if i == 0:
                s[i] = model.addVar(vtype='I', lb=0.0, ub=0, name="s(%s)" % i)  # fixed as in Tang et al.
            elif i == T:
                s[i] = model.addVar(vtype='I', lb=20, ub=20, name="s(%s)" % i)  # fixed as in Tang et al.
            else:
                s[i] = model.addVar(vtype='I', lb=0.0, ub=None, name="s(%s)" % i)

        for i in range(1, T+1):
            x[i] = model.addVar(vtype='I', lb=0.0, ub=None, name="x(%s)" % i)
            y[i] = model.addVar(vtype='B', lb=0.0, ub=1, name="y(%s)" % i)

            model.addCons(s[i-1] + x[i] == d[i] + s[i], "Demand(%s)" % i)
            model.addCons(x[i] <= M*y[i], "BigM(%s)" % i)

        model.setObjective(pyopt.quicksum(p[i-1]*x[i] for i in range(1, T+1)) +
                           pyopt.quicksum(h[i]*s[i] for i in range(T+1)) +
                           pyopt.quicksum(q[i-1]*y[i] for i in range(1, T+1)), "minimize")

        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data_dir', type=str,
                        default='../data',)
    args = parser.parse_args()
    main()
