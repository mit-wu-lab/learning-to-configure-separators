import numpy as np
import torch

CUT_IDENTIFIERS_TO_NUMS = {
    'cmir': 1,
    'flowcover': 2,
    'clique': 3,
    'dis': 4, #. ?
    'gom': 5,
    'implbd': 6,
    'mcf': 7,
    'oddcycle': 8,
    'scg': 9,
    'zerohalf': 10
}

SCIP_CUT_IDENTIFIERS_TO_NUMS = {
    # 'closecuts',
    'disjunctive': 1,
    # '#SM',
    # '#CS',
    'convexproj': 2,
    'gauge': 3,
    'impliedbounds': 4,
    'intobj': 5,
    'gomory': 6,
    'cgmip': 7,
    'strongcg': 8,
    'aggregation': 9,
    'clique': 10,
    'zerohalf': 11,
    'mcf': 12,
    'eccuts': 13,
    'oddcycle': 14,
    'flowcover': 15,
    'cmir': 16,
    'rapidlearning': 17
}

def computeSepas(sepa_states):
    res = [None for _ in range(len(sepa_states))]
    for sepa, v in sepa_states.items():
        res[CUT_IDENTIFIERS_TO_NUMS[sepa]-1] = v
    return res

def computeSCIPSepas(sepa_states):
    res = [None for _ in range(len(sepa_states))]
    for sepa, v in sepa_states.items():
        res[SCIP_CUT_IDENTIFIERS_TO_NUMS[sepa]-1] = v
    return res


def getCutTypeFromName(cut_name):

    for k, v in CUT_IDENTIFIERS_TO_NUMS.items():
        if k in cut_name:
            return v
    # unknown, return zero..
    return 0

def get_names(model):
    # we stick to the order of names this function returns
    row_names = []
    col_names = []
    cut_names = []

    for row in model.getLPRowsData():
        row_names.append(row.name)

    for col in model.getCols():
        col_names.append(col.name)

    for cut in model.getOptPoolCuts():
        cut_names.append(cut.name)

    return row_names, col_names, cut_names

def computeInputScores(cuts, model):
    score_funcs = [
        model.getCutViolation,
        model.getCutRelViolation,
        model.getCutObjParallelism,
        model.getCutEfficacy,
        # model.getCutDirectedCutoffDistance,
        # model.getCutAdjustedDirectedCutoffDistance,
        model.getCutSCIPScore,
        model.getCutExpImprov,
        model.getCutSupportScore,
        model.getCutIntSupport,
    ]

    scores = np.empty((len(cuts), len(score_funcs)), dtype=np.float32)
    for i, cut in enumerate(cuts):
        for j, score_func in enumerate(score_funcs):
            scores[i, j] = score_func(cut)

    return scores

def computeLookaheadScores(cuts, model):

    lpobjval = model.getLPObjVal()
    scores = np.empty([len(cuts), 3])
    scores[:, 0] =  lpobjval

    for (i, cut) in enumerate(cuts):
        scores[i, 1] = model.getCutLookaheadScore(cut)
        scores[i, 2] = model.getCutLookaheadLPObjval(cut)

    return scores


def computeSepaFeatures1(model, round_num=0):
    features = []
    stats = model.getSepaCumulatedStatics()
    for sepa_name, sepa_freq in SCIP_CUT_IDENTIFIERS_TO_NUMS.items():
        ft = {'#calls': stats[sepa_name]['time'], 'time': stats[sepa_name]['time'],
              '#cuts': stats[sepa_name]['#cuts'], '#cutoffs': stats[sepa_name]['#cutoffs'],
              '#applied': stats[sepa_name]['#applied'],
              'round_num': round_num, 'name': sepa_name}

        features.append(ft)
    return features


def computeRowFeatures1(rows, model, round_num=0):
    features = []
    for row in rows:
        ft = model.getRowFeatures1(row)
        ft['round_num'] = round_num
        features.append(ft)
    return features

def computeColFeatures1(cols, model, round_num=0):
    features = []
    for col in cols:
        ft = model.getColFeatures1(col)
        ft['round_num'] = round_num
        features.append(ft)
    return features

def computeCoefs(rows, cols, model):
    # hash col position for fast retrieval..
    col_dict = {}
    for j, col in enumerate(cols):
        colname = col.getVar().name
        assert not (colname in col_dict)
        col_dict[colname] = j

    coefs = {}
    for (i, row) in enumerate(rows):
        row_cols = row.getCols()
        row_js = [col_dict[col.getVar().name] for col in row_cols if col.getVar().name in col_dict]
        row_coefs = [val for val, col in zip(row.getVals(), row_cols) if col.getVar().name in col_dict]
        coefs[i] = (row_js, row_coefs)

    return coefs

def computeCutTypes(cuts):

    cut_types = np.empty((len(cuts), ), dtype=np.int32)

    for i, cut in enumerate(cuts):
        cut_types[i] = getCutTypeFromName(cut.name)

    return cut_types

def computeCutParallelism(cuts, model):

    cut_parallelism = []
    for i, cut1 in enumerate(cuts[:-1]): # exclude the very last
        for cut2 in cuts[(i+1):]:
            cut_parallelism.append(model.getRowParallelism(cut1, cut2))

    cut_parallelism = np.array(cut_parallelism, dtype=np.float32) #1D
    return cut_parallelism

def computeCutRowParallelism(cuts, rows, model):

    row_parallelism = np.empty((len(cuts), len(rows)), dtype=np.float32)

    for i, cut1 in enumerate(cuts): # exclude the very last  [:-1]
        for j, row in enumerate(rows):
            row_parallelism[i, j] = model.getRowParallelism(cut1, row)

    return row_parallelism