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

    single_effects = np.log(np.loadtxt(path + f"target1_{protein_concentrations[0]}_target2_{protein_concentrations[0]}/ground_truth.csv", delimiter=","))
    interactions = np.loadtxt(path + f"target1_{protein_concentrations[0]}_target2_{protein_concentrations[0]}/interaction_matrix.csv", delimiter=",")

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

    return single_effects, interactions, round_1_sequences, round_2_sequences, round_1_sequence_effects, round_2_sequence_effects, round_1_initial_frequencies, round_2_initial_frequencies, round_1_selected_frequencies, round_2_selected_frequencies

def infer_logK_sequences(initial_frequencies: np.array, selected_frequencies: np.array, total_protein_concentration: float, c: float, number_sequences: int = 1):

    # compute free protein concentration
    free_protein_concentration = selected_frequencies[0]/(initial_frequencies[0]-selected_frequencies[0])       #total_protein_concentration - np.sum(selected_frequencies/number_sequences)

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

    L = int(unique_sequences.shape[1])
    q = int(np.max(unique_sequences) + 1)
    number_mutations = int(L*(q-1))
    number_interactions = int(0.5 * L * (q-1) * (L*q - L - 3))

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
        mutations = x[:number_mutations]
        interactions = x[number_mutations:]

        # define A as the one-hot encoded sequence matrix
        A = one_hot_sequences

        # define X as the matrix of unknown single effects on hte diagonal
        X = np.zeros((number_mutations, number_mutations))
        np.fill_diagonal(X, mutations)

        # define E as the matrix of unknown interactions
        E = np.zeros((number_mutations, number_mutations))

        # fill E with the interactions
        i = 0
        for pos1 in range(L):
            for pos2 in range(pos1+1, L):
                for a1 in range(q-1):
                    for a2 in range(q-1):
                        E[pos1*(q-1)+a1, pos2*(q-1)+a2] = interactions[i]
                        i += 1
        # symmetrize
        E += E.T 
        #set diagonal to one
        np.fill_diagonal(E, 1)

        # compute regularization term for interactions
        # reg = 1e-3 * np.sum(np.square(interactions))

        return np.sum((np.sum((A @ X) * (A @ E), axis=1) - logK_sequences)**2) #+ reg
    
    #define initial guess
    x0 = np.zeros(int(number_mutations + number_interactions))

    #solve optimization problem
    result = sp.optimize.minimize(objective, x0)
    result.x[nan_indices] = np.nan

    print("\t" + result.message)

    single_effects = result.x[:number_mutations]
    # generate resulting interaction matrix
    interactions = np.zeros((number_mutations, number_mutations))
    i = 0
    for pos1 in range(L):
        for pos2 in range(pos1+1, L):
            for a1 in range(q-1):
                for a2 in range(q-1):
                    interactions[pos1*(q-1)+a1, pos2*(q-1)+a2] = result.x[number_mutations + i]
                    i += 1
    interactions += interactions.T
    np.fill_diagonal(interactions, 1)

    return single_effects, interactions

def logK_inference(path : str, protein_concentrations : list, c : float, number_sequences: int = 1):

    single_effects, interctions, round_1_sequences, round_2_sequences, round_1_sequence_effects, round_2_sequence_effects, round_1_initial_frequencies, round_2_initial_frequencies, round_1_selected_frequencies, round_2_selected_frequencies = get_pool_data(path, protein_concentrations)

    logK_sequences_r1 = []
    logK_mutations_r1 = []
    logK_sequences_r2 = []
    logK_mutations_r2 = []
    interactions_r1 = []
    interactions_r2 = []

    for i in range(len(protein_concentrations)):
        protein_concentration_1 = protein_concentrations[i]
        for j in range(len(protein_concentrations)):
            protein_concentration_2 = protein_concentrations[j]
            print(f"Pool {protein_concentration_1}, {protein_concentration_2}:")
            logK_sequences_r1.append(infer_logK_sequences(round_1_initial_frequencies[i*len(protein_concentrations) + j], round_1_selected_frequencies[i*len(protein_concentrations) + j], protein_concentration_1, c, number_sequences))
            logK_sequences_r2.append(infer_logK_sequences(round_2_initial_frequencies[i*len(protein_concentrations) + j], round_2_selected_frequencies[i*len(protein_concentrations) + j], protein_concentration_2, c, number_sequences))
            mutation_r1, interaction_r1 = infer_logK_mutations(logK_sequences_r1[-1], round_1_sequences[i*len(protein_concentrations) + j])
            mutation_r2, interaction_r2 = infer_logK_mutations(logK_sequences_r2[-1], round_2_sequences[i*len(protein_concentrations) + j])
            logK_mutations_r1.append(mutation_r1)
            logK_mutations_r2.append(mutation_r2)
            interactions_r1.append(interaction_r1)
            interactions_r2.append(interaction_r2)

    return logK_sequences_r1, logK_mutations_r1, interactions_r1, logK_sequences_r2, logK_mutations_r2, interactions_r2, single_effects, interctions, round_1_sequence_effects, round_2_sequence_effects
