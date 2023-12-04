import numpy as np
import states_helpers as _helpers

def getState(statestr, model, round_num=0):
    ### immediate steps
    # log_lookahead_score
    # lpobjval in state
    # lookahead num_branching candidates (for interest)
    # transform pickle -> numpy array to be more storage efficient

    # 1000 instances, 50 rounds, 1 cut per round (but code should support more cuts per round)
    ## let's assume 500 cuts per round (!),
    # 500 cuts per round, for parallelism give 0.5 MB
    # so 1000 instances give 0.5 GB for parallelism
    # for state, depends on size of instance,
    # we get 1000 x 50 x 1 states. so it'd be nice if it wasn't super large..

    if statestr == 'learn1':
        # should contain all data we want.
        ### assertion that col names are unique ###
        rows = model.getLPRowsData()
        cuts = model.getPoolCuts() + model.getCuts()
        cols = model.getLPColsData()

        row_features = _helpers.computeRowFeatures1(rows, model, round_num=round_num)
        col_features = _helpers.computeColFeatures1(cols, model, round_num=round_num)
        cut_features = _helpers.computeRowFeatures1(cuts, model, round_num=round_num)
        sepa_features = _helpers.computeSepaFeatures1(model, round_num=round_num)  

        state = {
            'cut_input_scores': _helpers.computeInputScores(cuts, model),
            'row_input_scores': _helpers.computeInputScores(rows, model),
            'cut_lookahead_scores': np.random.rand(len(cuts), 3),  # _helpers.computeLookaheadScores(cuts, model), -> this triggers some bug
            'row_features': row_features,
            'col_features': col_features,
            'cut_features': cut_features,
            'cut_parallelism': _helpers.computeCutParallelism(cuts, model),
            'cutrow_parallelism': _helpers.computeCutRowParallelism(cuts, rows, model),
            'row_coefs': _helpers.computeCoefs(rows, cols, model),
            'cut_coefs': _helpers.computeCoefs(cuts, cols, model),
            'sepa_features': sepa_features 
        }

    elif statestr == 'learn2':
        # should contain all data we want.
        ### assertion that col names are unique ###
        rows = model.getLPRowsData()
        cuts = model.getOptPoolCuts()
        cols = model.getLPColsData()

        state = {
            'cut_input_scores': _helpers.computeInputScores(cuts, model),
            'row_input_scores': _helpers.computeInputScores(rows, model),
            'cut_lookahead_scores': _helpers.computeLookaheadScores(cuts, model),
            'row_features': _helpers.computeRowFeatures1(rows, model),
            'col_features': _helpers.computeColFeatures1(cols, model),
            'cut_features': _helpers.computeRowFeatures1(cuts, model),
            'cut_parallelism': _helpers.computeCutParallelism(cuts, model),
            'cutrow_parallelism': _helpers.computeCutRowParallelism(cuts, rows, model),
            'row_coefs': _helpers.computeCoefs(rows, cols, model),
            'cut_coefs': _helpers.computeCoefs(cuts, cols, model),
        }

    elif statestr == 'scores':
        rows = model.getLPRowsData()
        cuts = model.getOptPoolCuts()

        state = {
            'cut_input_scores': _helpers.computeInputScores(cuts, model),
            'cut_lookahead_scores': _helpers.computeLookaheadScores(cuts, model),
            'cut_types': _helpers.computeCutTypes(cuts),
        }

    elif statestr == 'scores_parallelism':
        # this is the stuff we might want to consider for population-based
        # scoring only..

        rows = model.getLPRowsData()
        cuts = model.getOptPoolCuts()

        state = {
            'cut_input_scores': _helpers.computeInputScores(cuts, model),
            'cut_lookahead_scores': _helpers.computeLookaheadScores(cuts, model),
            'cut_types': _helpers.computeCutTypes(cuts),
            'cut_parallelism': _helpers.computeCutParallelism(cuts, model),
            'cutrow_parallelism': _helpers.computeCutRowParallelism(cuts, rows, model),
        }

    else:
        raise ValueError(f'Unknown state identifier: {statestr}')

    return state
