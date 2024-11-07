import numpy as np
import scipy as sp

path = '/datadisk/MIME/deterministic_prob_test/'
protein_concentrations = [0.1, 1, 10]
c = 0.0001

def get_pool_data(path : str, protein_concentrations : list):

    round_1_sequences = []
    round_2_sequences = []
    round_1_sequence_effects = []
    round_2_sequence_effects = []
    round_1_initial_frequencies = []
    round_2_initial_frequencies = []
    round_1_selected_frequencies = []
    round_2_selected_frequencies = []

    ground_truth = np.log(np.loadtxt(path + f"target1_{protein_concentrations[0]}_target2_{protein_concentrations[0]}/ground_truth.csv", delimiter=","))

    for protein_concentration_1 in protein_concentrations:
        for protein_concentration_2 in protein_concentrations:
            round_1_sequences.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_1/unique_sequences.csv", delimiter=","))
            round_2_sequences.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_2/unique_sequences.csv", delimiter=","))
            round_1_sequence_effects.append(np.log(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_1/sequence_effects.csv", delimiter=",")))
            round_2_sequence_effects.append(np.log(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_2/sequence_effects.csv", delimiter=",")))
            round_1_initial_frequencies.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_1/counts.csv", delimiter=","))
            round_2_initial_frequencies.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_2/counts.csv", delimiter=","))
            round_1_selected_frequencies.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_1/selected/counts.csv", delimiter=","))
            round_2_selected_frequencies.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_2/selected/counts.csv", delimiter=","))

    return ground_truth, round_1_sequences, round_2_sequences, round_1_sequence_effects, round_2_sequence_effects, round_1_initial_frequencies, round_2_initial_frequencies, round_1_selected_frequencies, round_2_selected_frequencies

def infer_logK_sequences(initial_frequencies: np.array, selected_frequencies: np.array, total_protein_concentration: float, c: float, number_sequences: int = 1):

    # compute free protein concentration
    free_protein_concentration = total_protein_concentration - np.sum(selected_frequencies/number_sequences)

    # get indices where selected frequency or initial frequency - selected frequency is less than c
    prune_indices = np.where((selected_frequencies < c) | (initial_frequencies - selected_frequencies < c))
    # print number of pruned sequences
    print(f"\tPruned {np.sum(initial_frequencies[prune_indices])} sequences")
    # set these indices to nan
    selected_frequencies[prune_indices] = np.nan
    initial_frequencies[prune_indices] = np.nan

    # compute K values of each sequence
    probability_selected = selected_frequencies/initial_frequencies
    K_sequence = (free_protein_concentration/probability_selected) - free_protein_concentration

    return np.log(K_sequence)

def one_hot_encoding(sequences):
    one_hot = np.zeros((sequences.shape[0], sequences.shape[1]*3))
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            if base == 0:
                continue
            elif base == 1:
                one_hot[i, j*3] = 1
            elif base == 2:
                one_hot[i, j*3+1] = 1
            elif base == 3:
                one_hot[i, j*3+2] = 1
    return one_hot

def infer_logK_mutations(logK_sequences : np.array, unique_sequences : np.array):

    # convert sequences to one-hot encoding
    one_hot_sequences = one_hot_encoding(unique_sequences)

    # get indices where K_sequences is nan
    prune_indices = np.where(np.isnan(logK_sequences))
    # remove these indices from one_hot_sequences and logK_sequences
    one_hot_sequences = np.delete(one_hot_sequences, prune_indices, axis=0)
    logK_sequences = np.delete(logK_sequences, prune_indices)
    # fill columns in one_hot_sequences that are all zeros with np.nan
    nan_indices = np.where(np.sum(one_hot_sequences, axis=0) == 0)

    #define optimization problem
    def objective(x):
        return np.sum((np.dot(one_hot_sequences, x) - logK_sequences)**2)
    
    #define initial guess
    x0 = np.zeros(one_hot_sequences.shape[1])

    #solve optimization problem
    result = sp.optimize.minimize(objective, x0)
    result.x[nan_indices] = np.nan

    print("\t" + result.message)

    return result.x

def logK_inference(path : str, protein_concentrations : list, c : float, number_sequences: int = 1):

    ground_truth, round_1_sequences, round_2_sequences, round_1_sequence_effects, round_2_sequence_effects, round_1_initial_frequencies, round_2_initial_frequencies, round_1_selected_frequencies, round_2_selected_frequencies = get_pool_data(path, protein_concentrations)

    logK_sequences_r1 = []
    logK_mutations_r1 = []
    logK_sequences_r2 = []
    logK_mutations_r2 = []

    for i in range(len(protein_concentrations)):
        protein_concentration_1 = protein_concentrations[i]
        for j in range(len(protein_concentrations)):
            protein_concentration_2 = protein_concentrations[j]
            print(f"Pool {protein_concentration_1}, {protein_concentration_2}:")
            logK_sequences_r1.append(infer_logK_sequences(round_1_initial_frequencies[i*len(protein_concentrations) + j], round_1_selected_frequencies[i*len(protein_concentrations) + j], protein_concentration_1, c, number_sequences))
            logK_mutations_r1.append(infer_logK_mutations(logK_sequences_r1[i*len(protein_concentrations) + j], round_1_sequences[i*len(protein_concentrations) + j]))
            logK_sequences_r2.append(infer_logK_sequences(round_2_initial_frequencies[i*len(protein_concentrations) + j], round_2_selected_frequencies[i*len(protein_concentrations) + j], protein_concentration_2, c, number_sequences))
            logK_mutations_r2.append(infer_logK_mutations(logK_sequences_r2[i*len(protein_concentrations) + j], round_2_sequences[i*len(protein_concentrations) + j]))

    return logK_sequences_r1, logK_mutations_r1, logK_sequences_r2, logK_mutations_r2, ground_truth, round_1_sequence_effects, round_2_sequence_effects
