import numpy as np
import pickle
import pandas as pd

from utils import *

actual_fitness = pickle.load(open("all_fitness.pkl", "rb"))

actual_fitness_temp = {i:[entry, actual_fitness[entry]] for i,entry in enumerate(actual_fitness)}

actual_fitness_df = pd.DataFrame.from_dict(actual_fitness_temp, orient='index')
actual_fitness_df.columns = ["AACombo", "Fitness"]

idx_to_combo_actual = pickle.load(open("idx_to_aacombo_map.pkl", "rb"))
combo_to_idx_actual = pickle.load(open("aacombo_to_idx_map.pkl", "rb"))

with open("./georgiev_enc/Encodings/example_protein_georgiev_ComboToIndex.pkl", "rb") as f:
    combo_to_idx_pred = pickle.load(f)
idx_to_combo_pred = {ind: combo for combo, ind in combo_to_idx_pred.items()}

base_1_combos = list(pd.read_csv("./Validation/BasicTestData/InputValidationData.csv").AACombo)
base_2_combos = list(np.load("base_2.npy"))


def combo_to_traindata(combo_list):
    training_data = [combo_list, [actual_fitness[combo] for combo in combo_list]]
    return pd.DataFrame(training_data, index=["AACombo", "Fitness"]).T


def select(all_results, train_pool, al_strategy, al_iter, num_select):
    """
    Filter model outputs, select num_select variants using al_strategy and add to train_pool
    """
    train_error, test_error, pred_fitness, pred_fitness_std = all_results

    train_data_pred, train_data_std = filter(pred_fitness, pred_fitness_std, train_pool)

    if al_strategy[al_iter] == "ucb":
        new_combos = upper_confidence_bound(train_data_pred, train_data_std, num_select)
    elif al_strategy[al_iter] == "lcb":
        new_combos = upper_confidence_bound(train_data_pred, train_data_std, num_select)
    elif al_strategy[al_iter] == "weighted_ucb":
        new_combos = upper_confidence_bound(train_data_pred, train_data_std, num_select, uncertainty = False, weighted = True, model_test_error = test_error)
    elif al_strategy[al_iter] == "weighted_lcb":
        new_combos = lower_confidence_bound(train_data_pred, train_data_std, num_select, uncertainty = False, weighted = True, model_test_error = test_error)
    elif al_strategy[al_iter] == "random":
        new_combos = random(list(train_data_pred.keys()), num_select)
    elif al_strategy[al_iter] == "ddG":
        new_combos = ddG(list(train_data_pred.keys()), num_select)
    elif al_strategy[al_iter] == "variance":
        new_combos = variance(train_data_pred, train_data_std, num_select)
    elif al_strategy[al_iter] == "384":
        train_data_pred = {combo:train_data_pred[combo] for combo in train_data_pred if combo in base_1_combos}
        print(len(train_data_pred))
        new_combos = random(list(train_data_pred.keys()), num_select)
    elif al_strategy[al_iter] == "384b":
        train_data_pred = {combo:train_data_pred[combo] for combo in train_data_pred if combo in base_2_combos}
        print(len(train_data_pred))
        new_combos = random(list(train_data_pred.keys()), num_select)
    else:
        raise ValueError("{} AL strategy not implemented".format(al_strategy))

    if len(list(set(new_combos) & set(train_pool))) > 0:
        raise ValueError("Adding redundant data to train_pool")
    train_pool = train_pool + new_combos

    return train_pool

def filter(pred_fitness, pred_fitness_std, train_pool):
    """
    Filter out variants we don't have fitness values for and are not already in train pool
    """
    # combo:predicted fitness for all models
    train_data_pred = {}

    # combo:predicted fitness std for all models
    train_data_std = {}

    for i in range(160000):
        train_data_pred[idx_to_combo_pred[i]] = pred_fitness[:,i]

    for i in range(160000):
        train_data_std[idx_to_combo_pred[i]] = pred_fitness_std[:,i]

    # remove combos we do not have fitness values for
    train_data_pred = {combo:train_data_pred[combo] for combo in train_data_pred if combo in combo_to_idx_actual.keys()}
    train_data_std = {combo:train_data_std[combo] for combo in train_data_pred if combo in combo_to_idx_actual.keys()}

    # remove combos that are in train pool
    train_data_pred = {combo:train_data_pred[combo] for combo in train_data_pred if combo not in train_pool}
    train_data_std = {combo:train_data_std[combo] for combo in train_data_pred if combo not in train_pool}

    return train_data_pred, train_data_std

def prediction_error(pred_fitness_df):
    # filter predicted fitness that we don't have measurements for and we did not train on
    merged_fitness_df = pd.merge(actual_fitness_df, pred_fitness_df, how='inner', on=['AACombo'])
    merged_fitness_df = merged_fitness_df[merged_fitness_df["Train"] == 0]

    pred_fitness = merged_fitness_df.PredictedFitness.to_numpy()
    actual_fitness = merged_fitness_df.Fitness.to_numpy()

    mse = avg_mean_squared_error(actual_fitness, pred_fitness)
    # ndcg_val = ndcg(actual_fitness, pred_fitness)
    max_m = max_m_fitness(actual_fitness, pred_fitness)
    mean_m = mean_m_fitness(actual_fitness, pred_fitness)

    return mse, max_m, mean_m


def vis_train_pool(train_pool):
    train_pool_indicator = np.zeros(149361)
    for combo in train_pool:
        train_pool_indicator[combo_to_idx_actual[combo]] = 1

    return train_pool_indicator

def intersect_train_pool(tp1, tp2):
    return list(set(tp1) & set(tp2))

def fitness_train_pool(train_pools):
    """
    Get actual fitness values of elements in the train pool
    """
    all_combos = []
    for tp in train_pools:
        all_combos += tp
    return [actual_fitness[combo] for combo in all_combos]

def fitness_variants(variants):
    """
    Get actual fitness values of variants
    """
    return [actual_fitness[combo] for combo in variants]
