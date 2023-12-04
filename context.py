import numpy as np
import utils as _utils
import data as _data
import states_helpers as _helpers
import torch
import torch.nn as nn
import pyscipopt as pyopt

path_to_randomness_control_set = './SCIP_settings/randomness_control.set'

def getActionContext(statestr, input_context, action):
    if statestr == 'bandit':
        action_context = input_context
        action_context.x_sepas = torch.tensor(action, dtype=torch.float)
    
    else:
        raise ValueError(f'Unknown state identifier: {statestr}')

    return action_context

def getUpdatedInputContext(model, instance_type, cuts=None, round_num=0, max_sepa_round_num=10, mask=None):
    rows = model.getLPRowsData()
    cols = model.getLPColsData()

    if type(cuts) != type(None):
        n_cuts = len(cuts)
    else:
        n_cuts = 0
    n_row = len(rows)
    n_cols = len(cols)

    # the model might not use all of this, in this case, we just create
    # mock inputs, so we don't have to rewrite the rest of the code.
    # this makes good sense, even though it is computationally stupid.

    cut_input_scores = _helpers.computeInputScores(cuts, model)

    row_input_scores = _helpers.computeInputScores(rows, model)

    # cut_lookahead_scores = _helpers.computeLookaheadScores(cuts, model)
    cut_lookahead_scores = np.ones((n_cuts, 3))
    cut_lookahead_scores[:, 2] = 2  # not sure if there were any checks for consistency

    row_features = _helpers.computeRowFeatures1(rows, model, round_num=round_num / max_sepa_round_num)
    col_features = _helpers.computeColFeatures1(cols, model, round_num=round_num / max_sepa_round_num)
    cut_features = _helpers.computeRowFeatures1(cuts, model, round_num=round_num / max_sepa_round_num)

    cut_parallelism = _helpers.computeCutParallelism(cuts, model)

    cutrow_parallelism = _helpers.computeCutRowParallelism(cuts, rows, model)

    row_coefs = _helpers.computeCoefs(rows, cols, model)

    # dictionary, int: tuple(list, list)
    # key is position of cut, then ixs, vals
    cut_coefs = _helpers.computeCoefs(cuts, cols, model)

    sepa_settings = np.ones(len(_helpers.SCIP_CUT_IDENTIFIERS_TO_NUMS))

    sepa_features = _helpers.computeSepaFeatures1(model, round_num=round_num / max_sepa_round_num)

    raw_data = {
        'cut_input_scores.npy': cut_input_scores,
        'row_input_scores.npy': row_input_scores,
        'cut_lookahead_scores.npy': cut_lookahead_scores,
        'row_features.pkl': row_features,
        'col_features.pkl': col_features,
        'cut_features.pkl': cut_features,
        'cut_parallelism.npy': cut_parallelism,
        'cutrow_parallelism.npy': cutrow_parallelism,
        'row_coefs.pkl': row_coefs,
        'cut_coefs.pkl': cut_coefs,
        'sepa_settings.pkl': sepa_settings,
        'masks.npy': np.zeros(len(sepa_settings)) if mask is None else mask,
        'sepa_features.pkl': sepa_features
    }

    min_parall = _data.MyData.get_minparallelism(instance_type)
    maxnum_cutixs, maxnum_rowixs, maxnum_rowcolixs = _data.MyData.get_maxnums(instance_type)
    processed_data =_data.MyData.from_rawdata(raw_data, min_parall, maxnum_cutixs, maxnum_rowixs, maxnum_rowcolixs)
    inp = _data.MyData(*processed_data)
    return inp

class SepaManager_SM_Context(pyopt.Sepa):
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
        self.sepa_round += 1
        if self.sepa_round == self.context_step:
            self.contextAtStep = getUpdatedInputContext(
                self.model, 
                self.instance_type,
                [],
                round_num=self.context_step,
            )
            self.model.interruptSolve()
            self.interrupt_flag = True
            return {'result': pyopt.SCIP_RESULT.DIDNOTRUN}
        if self.interrupt_flag == True:
            return {'result': pyopt.SCIP_RESULT.DIDNOTRUN}
        
        if self.sepa_round in self.orders:
            for index in range(len(self.sepa_list)):
                on_or_off = 1
                # data collection: search trajectory
                if self.orders[self.sepa_round][index,0] == 0:
                    on_or_off = -1
                self.model.setParam(f'separating/{self.sepa_list[index]}/freq', on_or_off)
        return {'result': pyopt.SCIP_RESULT.DIDNOTRUN}

    def sepaexecsol(self):
        # actual behaviour is implemented in method self.main
        return {'result': pyopt.SCIP_RESULT.DIDNOTRUN}

    def addModel(self, model, orders={}, context_step=-1, instance_type="packing-60-60"):
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
        self.model.setParam(f'separating/#SM/freq', 1)
        self.model.setParam(f'separating/closecuts/freq', -1)
        self.sepa_round = 0
        self.orders = orders
        self.context_step = context_step
        self.interrupt_flag = False
        self.contextAtStep = None
        self.instance_type = instance_type
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
        assert self.model == model  # PySCIPOpt sets that in pyopt.Sepa
    
    def get_sepa_round(self):
        return self.sepa_round
    
    def get_context(self):
        return self.contextAtStep
    
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

def getInputContextAtStep(action_step0, path_to_problem, step, instance_type, gap_limit=0.00):
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

    model = pyopt.Model()
    model.hideOutput(1)
    model.readProblem(path_to_problem)
    model.readParams(path_to_randomness_control_set)
    model.setRealParam('limits/gap', gap_limit)
    if type(action_step0) != type(None):
        for index in range(len(sepa_list)):
            on_or_off = 1
            # data collection: search trajectory
            if action_step0[index,0] == 0:
                on_or_off = -1
            model.setParam(f'separating/{sepa_list[index]}/freq', on_or_off)
    sepa_manager_SM = SepaManager_SM_Context()
    sepa_manager_SM.addModel(model, context_step=step, instance_type=instance_type)
    model.optimize()
    return sepa_manager_SM.get_context()

class SepaManager_SM_Context_Actions(pyopt.Sepa):
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
        self.sepa_round += 1
        if self.sepa_round == self.context_step:
            self.contextAtStep = getUpdatedInputContext(
                self.model, 
                self.instance_type,
                [],
                round_num=self.context_step,
            )
            self.model.interruptSolve()
            self.interrupt_flag = True
            return {'result': pyopt.SCIP_RESULT.DIDNOTRUN}
        if self.interrupt_flag == True:
            return {'result': pyopt.SCIP_RESULT.DIDNOTRUN}
        
        for Bandit, bandit_step in self.Bandits:
            if self.sepa_round == bandit_step:
                cur_input_context = getUpdatedInputContext(
                    self.model, 
                    self.instance_type,
                    [],
                    round_num=self.sepa_round,
                )
                action_tmp, _ = Bandit.getActions(cur_input_context, 1, eva=1)
                action = action_tmp[0]
                for index in range(len(self.sepa_list)):
                    on_or_off = 1
                    # data collection: search trajectory
                    if action[index, 0] < 0.5:
                        on_or_off = -1
                    self.model.setParam(f'separating/{self.sepa_list[index]}/freq', on_or_off)
                self.actions[self.sepa_round] = action
        return {'result': pyopt.SCIP_RESULT.DIDNOTRUN}

    def sepaexecsol(self):
        # actual behaviour is implemented in method self.main
        return {'result': pyopt.SCIP_RESULT.DIDNOTRUN}

    def addModel(self, model, Bandits, context_step, instance_type):
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
        self.model.setParam(f'separating/#SM/freq', 1)
        self.model.setParam(f'separating/closecuts/freq', -1)
        self.sepa_round = -1
        self.Bandits = Bandits
        self.context_step = context_step
        self.interrupt_flag = False
        self.contextAtStep = None
        self.instance_type = instance_type
        self.actions = {}
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
        assert self.model == model  # PySCIPOpt sets that in pyopt.Sepa
    
    def get_sepa_round(self):
        return self.sepa_round
    
    def get_context(self):
        return self.contextAtStep
    
    def get_actions(self):
        return self.actions
    
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

def getInputContextAndInstruction(path_to_problem, step, instance_type, Bandits, gap_limit=0.00):
    model = pyopt.Model()
    model.hideOutput(1)
    model.readProblem(path_to_problem)
    model.readParams(path_to_randomness_control_set)
    model.setRealParam('limits/gap', gap_limit)
    sepa_manager_SM = SepaManager_SM_Context_Actions()
    sepa_manager_SM.addModel(model, Bandits=Bandits, context_step=step, instance_type=instance_type)
    model.optimize()
    return sepa_manager_SM.get_context(), sepa_manager_SM.get_actions()