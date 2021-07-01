import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import ndcg_score

"""
Querying strategies and performance metrics
"""

def upper_confidence_bound(pred_fitness, pred_fitness_std, num_select, uncertainty = False, weighted = False, model_test_error = None):
    if weighted:
        #TODO: weight contribution of model based on model test error
        #normalize model_test_error and multiply rows of pred_fitness by that normalized value
        combo_ucb = {combo:pred_fitness[combo][:5].mean() + 2 * pred_fitness[combo][:5].std() for combo in pred_fitness.keys()}
    elif uncertainty:
        combo_ucb = {combo:(pred_fitness[combo].mean() + 2 * pred_fitness_std[combo]) + pred_fitness[combo].std() * 2 for combo in pred_fitness.keys()}
    else:
        combo_ucb = {combo:pred_fitness[combo].mean() + (pred_fitness[combo].std() * 2) for combo in pred_fitness.keys()}

    new_combos = [k for k, v in sorted(combo_ucb.items(), key=lambda item: item[1])]
    new_combos_vals = [v for k, v in sorted(combo_ucb.items(), key=lambda item: item[1])]

    return new_combos[-num_select:]

def lower_confidence_bound(pred_fitness, pred_fitness_std, num_select, uncertainty = False, weighted = False, model_test_error = None):
    if weighted:
        combo_ucb = {combo:pred_fitness[combo][:5].mean() - (pred_fitness[combo][:5].std() * 2) for combo in pred_fitness.keys()}
    elif uncertainty:
        combo_ucb = {combo:(pred_fitness[combo].mean() + 2 * pred_fitness_std[combo]) - pred_fitness[combo].std() * 2 for combo in pred_fitness.keys()}
    else:
        combo_ucb = {combo:pred_fitness[combo].mean() - (pred_fitness[combo].std() * 2) for combo in pred_fitness.keys()}

    new_combos = [k for k, v in sorted(combo_ucb.items(), key=lambda item: item[1])]
    return new_combos[-num_select:]

def variance(pred_fitness, pred_fitness_std, num_select, model_train_error = None, model_test_error = None):
    combo_ucb = {combo:pred_fitness[combo][:5].std() for combo in pred_fitness.keys()}
    new_combos = [k for k, v in sorted(combo_ucb.items(), key=lambda item: item[1])]

    return new_combos[-num_select:]

def random(combos, num_select):
    return list(np.random.choice(combos, num_select))

def ddG(combos, num_select):
    pass

def ndcg(actual_fitness, pred_fitness):
    return ndcg_score(actual_fitness, pred_fitness)

def max_m_fitness(actual_fitness, pred_fitness, m = 96):
    assert m <= len(actual_fitness)
    top_m_idxs = pred_fitness.argsort()[-m:][::-1]
    return actual_fitness[top_m_idxs].max()/8.762

def mean_m_fitness(actual_fitness, pred_fitness, m = 96):
    assert m <= len(actual_fitness)
    top_m_idxs = pred_fitness.argsort()[-m:][::-1]
    return actual_fitness[top_m_idxs].mean()/8.762

def avg_mean_squared_error(actual_fitness, pred_fitness):
    return np.mean((actual_fitness - pred_fitness)**2)
