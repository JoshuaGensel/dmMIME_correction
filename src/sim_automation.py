import stochastic_MIME_simulator
import MIME_correction
import numpy as np

def get_pool_data(name : str, first_round_targets : str, second_round_targets : str):
    path = f"/datadisk/MIME/{name}/"
    ground_truth = np.loadtxt(path + f"target1_{first_round_targets}_target2_{second_round_targets}/ground_truth.csv", delimiter=",")
    round_1 = np.log(np.loadtxt(path + f"target1_{first_round_targets}_target2_{second_round_targets}/round_1/effects.csv", delimiter=","))
    round_2 = np.log(np.loadtxt(path + f"target1_{first_round_targets}_target2_{second_round_targets}/round_2/effects.csv", delimiter=","))

    path_unbound_pairwise_counts_r1 = path + f"target1_{first_round_targets}_target2_{second_round_targets}/round_1/non_selected/pairwise_count.csv"
    path_unbound_pairwise_counts_r2 = path + f"target1_{first_round_targets}_target2_{second_round_targets}/round_2/non_selected/pairwise_count.csv"
    path_bound_pairwise_counts_r1 = path + f"target1_{first_round_targets}_target2_{second_round_targets}/round_1/selected/pairwise_count.csv"
    path_bound_pairwise_counts_r2 = path + f"target1_{first_round_targets}_target2_{second_round_targets}/round_2/selected/pairwise_count.csv"

    frequency_matrix_r1 = MIME_correction.construct_frequency_matrix(path_unbound_pairwise_counts_r1, path_bound_pairwise_counts_r1)
    frequency_matrix_r2 = MIME_correction.construct_frequency_matrix(path_unbound_pairwise_counts_r2, path_bound_pairwise_counts_r2)

    corrected_round_1 = np.linalg.solve(frequency_matrix_r1, round_1)
    corrected_round_2 = np.linalg.solve(frequency_matrix_r2, round_2)

    return ground_truth, round_1, round_2, corrected_round_1, corrected_round_2

def get_experiment_data(name :str):
    ground_truths = []
    round_1s = []
    round_2s = []
    corrected_round_1s = []
    corrected_round_2s = []

    for first_round_targets in ["0.1", "1", "10"]:
        for second_round_targets in ["0.1", "1", "10"]:
            ground_truth, round_1, round_2, corrected_round_1, corrected_round_2 = get_pool_data(name, first_round_targets, second_round_targets)
            ground_truths.append(ground_truth)
            round_1s.append(round_1)
            round_2s.append(round_2)
            corrected_round_1s.append(corrected_round_1)
            corrected_round_2s.append(corrected_round_2)

    return ground_truth, round_1s, round_2s, corrected_round_1s, corrected_round_2s

def fit_slopes(ground_truth, round_1s, round_2s, corrected_round_1s, corrected_round_2s):
    slopes_r1 = []
    corrected_slopes_r1 = []

    for i in range(9):
        #fit slopes for ground truth vs round_1s with np.polyfit
        slope, _ = np.polyfit(ground_truth, round_1s[i], 1)
        slopes_r1.append(slope)

        #fit slopes for ground truth vs corrected_round_1s with np.polyfit
        corrected_slope, _ = np.polyfit(ground_truth, corrected_round_1s[i], 1)
        corrected_slopes_r1.append(corrected_slope)

    slopes_r2 = []
    corrected_slopes_r2 = []

    for i in range(9):
        #fit slopes for ground truth vs round_2s with np.polyfit
        slope, _ = np.polyfit(ground_truth, round_2s[i], 1)
        slopes_r2.append(slope)

        #fit slopes for ground truth vs corrected_round_2s with np.polyfit
        corrected_slope, _ = np.polyfit(ground_truth, corrected_round_2s[i], 1)
        corrected_slopes_r2.append(corrected_slope)

    return slopes_r1, corrected_slopes_r1, slopes_r2, corrected_slopes_r2

def depth_test():
    lengths = [3,5,7,10,15,20,50]
    depths = [1000,10000,100000,1000000]