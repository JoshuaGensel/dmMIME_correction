import numpy as np
import scipy as sp
from scipy import stats
from tqdm import tqdm

path = '/datadisk/MIME/deterministic_prob_test/'
protein_concentrations = [0.1, 1, 10]
c = 0.0001

def get_pool_data_sim(path : str, protein_concentrations : list):

    round_1_sequences = []
    round_2_sequences = []
    round_1_sequence_effects = []
    round_2_sequence_effects = []
    round_1_initial_frequencies = []
    round_2_initial_frequencies = []
    round_1_selected_frequencies = []
    round_2_selected_frequencies = []
    round_1_nonselected_frequencies = []
    round_2_nonselected_frequencies = []

    single_effects = np.log(np.loadtxt(path + f"target1_{protein_concentrations[0]}_target2_{protein_concentrations[0]}/ground_truth.csv", delimiter=","))
    interactions = np.loadtxt(path + f"target1_{protein_concentrations[0]}_target2_{protein_concentrations[0]}/interaction_matrix.csv", delimiter=",")
    error_rates = np.loadtxt(path + "error_rates.csv")

    for protein_concentration_1 in protein_concentrations:
        for protein_concentration_2 in protein_concentrations:
            round_1_sequences.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_1/unique_sequences.csv", delimiter=","))
            round_2_sequences.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_2/unique_sequences.csv", delimiter=","))
            round_1_sequence_effects.append(np.log(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_1/sequence_effects.csv", delimiter=",")))
            round_2_sequence_effects.append(np.log(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_2/sequence_effects.csv", delimiter=",")))
            round_1_initial_frequencies.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_1/counts.csv", delimiter=","))
            round_2_initial_frequencies.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_2/counts.csv", delimiter=","))
            round_1_selected_frequencies.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_1/selected/counts_with_error.csv", delimiter=","))
            round_2_selected_frequencies.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_2/selected/counts_with_error.csv", delimiter=","))
            round_1_nonselected_frequencies.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_1/non_selected/counts_with_error.csv", delimiter=","))
            round_2_nonselected_frequencies.append(np.loadtxt(path + f"target1_{protein_concentration_1}_target2_{protein_concentration_2}/round_2/non_selected/counts_with_error.csv", delimiter=","))

    return single_effects, interactions, error_rates, round_1_sequences, round_2_sequences, round_1_sequence_effects, round_2_sequence_effects, round_1_initial_frequencies, round_2_initial_frequencies, round_1_selected_frequencies, round_2_selected_frequencies, round_1_nonselected_frequencies, round_2_nonselected_frequencies

def get_pool_data_exp(path : str, protein_concentrations : list):

    round_2_sequences = []
    round_2_selected_frequencies = []
    round_2_nonselected_frequencies = []
    round_2_initial_frequencies = []

    error_rates = np.loadtxt(path + "encoded_wt_mean_error_probs.txt")
    significant_positions = np.loadtxt(path + "sig_pos2.txt")
    data_path = path + "parsed_data2/round2/"

    for protein_concentration_1 in protein_concentrations:
        for protein_concentration_2 in protein_concentrations:
            round_2_sequences.append(np.genfromtxt(data_path + f"encoded_pool_{protein_concentration_1}_{protein_concentration_2}/encoded_pool/unique_sequences.txt", delimiter=1))

            nonselected_frequencies = np.loadtxt(data_path + f"encoded_pool_{protein_concentration_1}_{protein_concentration_2}/encoded_pool/unbound_counts.txt")
            round_2_nonselected_frequencies.append(nonselected_frequencies)
            
            selected_frequencies = np.loadtxt(data_path + f"encoded_pool_{protein_concentration_1}_{protein_concentration_2}/encoded_pool/bound_counts.txt")
            round_2_selected_frequencies.append(selected_frequencies)
            
            # initial frequencies are the sum of the selected and nonselected frequencies
            initial_frequencies = np.sum([selected_frequencies, nonselected_frequencies], axis=0)
            round_2_initial_frequencies.append(initial_frequencies)
            
    return round_2_sequences, round_2_initial_frequencies, round_2_selected_frequencies, round_2_nonselected_frequencies, error_rates, significant_positions


# get_pool_data_exp('/datadisk/MIME/exp/expData/', [8])



def correct_sequencing_error(sequences : np.array, counts : np.array, error_rates : np.array, number_states : int = 4, method : str = 'sampling'):
    # check that method is valid
    if method not in ('sampling', 'inversion', 'none'):
        raise ValueError("invalid correction method")
    
    if method == 'sampling':

        def add_sequence_error(sequences, counts, sequencing_error_rates):
    
            full_sequences = np.repeat(sequences, np.round(np.maximum(counts, 0)).astype(int), axis=0)
            # add errors per position according to the error rates
            for i in range(full_sequences.shape[1]):
                # draw if error occurs
                error = np.random.binomial(1, sequencing_error_rates[i], full_sequences.shape[0])
                # draw the error as uniform over the states
                error_states = np.random.randint(1, number_states, full_sequences.shape[0], dtype=np.ubyte)
                # apply the error
                full_sequences[:, i] = full_sequences[:, i] * (1 - error) + error_states * error
            # make sequences unique
            sequences_with_errors, counts_with_errors = np.unique(full_sequences, axis=0, return_counts=True)

            sequences_with_errors_matched = []
            counts_matched = []

            #match the error sequences with the original sequences
            for seq in range(sequences.shape[0]):
                # find the matching sequences
                matching = np.where(np.all(sequences[seq] == sequences_with_errors, axis=1))[0]
                # if there are matching sequences, add them to the list
                if len(matching) > 0:
                    sequences_with_errors_matched.append(sequences_with_errors[matching][0])
                    counts_matched.append(counts_with_errors[matching][0])
                # if there are no matching sequences, add the original sequence with 0 counts
                else:
                    sequences_with_errors_matched.append(sequences[seq])
                    counts_matched.append(0)
            # convert to numpy arrays
            sequences_with_errors_matched = np.array(sequences_with_errors_matched)
            counts_matched = np.array(counts_matched)
                    

            return sequences_with_errors, counts_with_errors, sequences_with_errors_matched, counts_matched
        

        # initial parameters are the observed counts
        n_iter = 100
        kappa = 10
        parameters = counts
        candidates = []
        best_kl = np.inf
        best_counts = parameters
        kl_list = []
        # apply the error function to the initial distribution
        for i in tqdm(range(n_iter)):
            sequences_with_error, counts_with_error, sequences_with_errors_matched, counts_matched = add_sequence_error(sequences, parameters, error_rates)

            # compute KL divergence
            kl = stats.entropy(counts+1e-10, counts_matched+1e-10)
            kl_list.append(kl)
            # if the KL divergence is smaller than the best KL divergence, update the best KL divergence
            if kl < best_kl:
                best_kl = kl
                best_counts = parameters
            # update the initial distribution
            parameters = np.maximum(0,parameters + (counts - counts_matched) + np.random.randint(-2, 2, counts.shape))
            candidates.append(parameters)

        # print(kl_list)
        # check that weights don't sum to 0
        # if np.sum(1/(np.array(kl_list)+1)) == 0:
            # return best_counts
        mean_parameters = np.round(np.average(candidates, axis=0, weights=1/(kappa*np.array(kl_list)+1)))
        
        return mean_parameters
    
    elif method == 'inversion':
        
        # initial transition matrix
        transition_matrix = np.zeros((sequences.shape[0], sequences.shape[0]))
        # fill the transition matrix with the number of different elements in the sequences
        for i in tqdm(range(sequences.shape[0])):
            for j in range(sequences.shape[0]):
                if i == j:
                    continue
                position_wise_probability = []
                for k in range(sequences.shape[1]):
                    if sequences[i, k] != sequences[j, k]:
                        position_wise_probability.append(error_rates[k]/(number_states - 1))
                    else:
                        position_wise_probability.append(1 - error_rates[k])
                transition_matrix[i, j] = np.prod(position_wise_probability)

        # set diagonal to (1 - sum of row)
        transition_matrix = transition_matrix + np.diag(1 - np.sum(transition_matrix, axis=1))
        # print(np.min(transition_matrix))
        # print(np.sum(transition_matrix, axis=1))

        # check if matrix is invertible
        # if np.linalg.matrix_rank(transition_matrix) < transition_matrix.shape[0]:
            # print("Matrix is not invertible")
            # return None

        # compute the inverse transition matrix
        inverse_transition_matrix = np.linalg.inv(transition_matrix)

        # normalize counts
        counts_norm = counts / np.sum(counts)
        # print(counts_norm)
        # compute the full inversion
        full_inversion = np.dot(counts_norm, inverse_transition_matrix)
        # print(full_inversion)

        # # make sure the full inversion is positive
        # full_inversion = np.maximum(0, full_inversion)

        # # normalize the full inversion
        # full_inversion = full_inversion / np.sum(full_inversion)

        #scale and round the full inversion
        full_inversion = np.round(full_inversion * np.sum(counts))
        # set minimum count to 0
        full_inversion = np.maximum(0.0, full_inversion)

        return full_inversion
    
    elif method == 'none':
        return counts

def infer_logK_sequences(unique_sequences: np.array, initial_frequencies: np.array, selected_frequencies_w_error: np.array, nonselected_frequencies_w_error: np.array, error_rates : np.array, protein_concentration: float, c: float, correction_method: str = 'sampling', max_mutations: int = 2):

    # correct selected and nonselected frequencies
    selected_frequencies = correct_sequencing_error(unique_sequences, selected_frequencies_w_error, error_rates, method=correction_method)
    nonselected_frequencies = correct_sequencing_error(unique_sequences, nonselected_frequencies_w_error, error_rates, method=correction_method)

    # compute free protein concentration
    free_protein_concentration = selected_frequencies[0]/(nonselected_frequencies[0])       #total_protein_concentration - np.sum(selected_frequencies/number_sequences)

    # get indices where selected frequency or initial frequency - selected frequency is less than c
    prune_indices_c = np.where((selected_frequencies < c) | (nonselected_frequencies < c))
    # get indices where number of mutations (nonzero elements) is greater than max_mutations
    prune_indices_m = np.where(np.sum(unique_sequences != 0, axis=1) > max_mutations)
    # combine indices
    prune_indices = np.unique(np.concatenate((prune_indices_c[0], prune_indices_m[0])))
    # print number of pruned sequences
    print(f"\tPruned {np.sum(initial_frequencies[prune_indices])} of {np.sum(initial_frequencies)} sequences")
    # set these indices to nan
    selected_frequencies[prune_indices] = np.nan
    initial_frequencies[prune_indices] = np.nan

    # compute K values of each sequence
    probability_selected = selected_frequencies/(selected_frequencies + nonselected_frequencies)
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

def infer_logK_mutations(logK_sequences : np.array, unique_sequences : np.array, lambda_l1 : float = 0.001):

    L = int(unique_sequences.shape[1])
    q = int(np.max(unique_sequences) + 1)
    number_mutations = int(L*(q-1))

    # convert sequences to one-hot encoding
    one_hot_sequences = one_hot_encoding(unique_sequences)
    # get indices where K_sequences is nan
    prune_indices = np.where(np.isnan(logK_sequences))
    # remove these indices from one_hot_sequences and logK_sequences
    one_hot_sequences = np.delete(one_hot_sequences, prune_indices, axis=0)
    logK_sequences = np.delete(logK_sequences, prune_indices)
    # fill columns in one_hot_sequences that are all zeros with np.nan
    nan_indices = np.where(np.sum(one_hot_sequences, axis=0) == 0)

    # get mutation pairs that the interaction matrix will be inferred for
    mutation_pairs = []
    for i in range(L*(q-1)):
        for j in range(i+1, L*(q-1)):
            # check if there is at least one sequence with a mutation at both positions
            if np.sum(one_hot_sequences[:, i] * one_hot_sequences[:, j]) >= 1:
                # also check if there is at least one sequence with a mutation at only one of the positions
                if np.sum(one_hot_sequences[:, i]) > np.sum(one_hot_sequences[:, i] * one_hot_sequences[:, j]) and np.sum(one_hot_sequences[:, j]) > np.sum(one_hot_sequences[:, i] * one_hot_sequences[:, j]):
                    mutation_pairs.append((i, j))

    number_interactions = len(mutation_pairs)    

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
        for pair in mutation_pairs:
            E[pair[0], pair[1]] = interactions[i]
            i += 1

        # symmetrize
        E += E.T 
        #set diagonal to one
        np.fill_diagonal(E, 1)

        # compute regularization term for interactions
        reg = lambda_l1 * np.sum(np.abs(interactions))
        return np.sum((np.sum((A @ X) * (A @ E), axis=1) - logK_sequences)**2) + reg
    
    #define initial guess
    x0 = np.zeros(int(number_mutations + number_interactions))

    # define bounds for interactions
    bounds = []
    for i in range(number_mutations):
        bounds.append((-10, 10))
    for i in range(number_interactions):
        bounds.append((-2, 2))
    #solve optimization problem
    result = sp.optimize.minimize(objective, x0, bounds=bounds, method='L-BFGS-B', options={'maxfun': 1000000, 'maxiter': 100000})
    result.x[nan_indices] = np.nan

    print("\t" + result.message)

    single_effects = result.x[:number_mutations]
    # generate resulting interaction matrix
    interactions = np.zeros((number_mutations, number_mutations))
    i = 0
    for pair in mutation_pairs:
        interactions[pair[0], pair[1]] = result.x[number_mutations + i]
        i += 1

    interactions += interactions.T
    np.fill_diagonal(interactions, 1)

    # set all zeros in interactions to nan
    interactions[interactions == 0] = np.nan

    print(f"\t number of possible interactions: {0.5 * L * (q-1) * (L*q - L - 3)}")
    print(f"\t number of inferred interactions: {number_interactions}")

    return single_effects, interactions

def logK_inference_sim(path : str, protein_concentrations : list, c : float, lambda_l1 : float = 0.001, correction_method: str = 'sampling'):

    single_effects, interctions, error_rates, round_1_sequences, round_2_sequences, round_1_sequence_effects, round_2_sequence_effects, round_1_initial_frequencies, round_2_initial_frequencies, round_1_selected_frequencies, round_2_selected_frequencies, round_1_nonselected_frequencies, round_2_nonselected_frequencies = get_pool_data_sim(path, protein_concentrations)

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
            logK_sequences_r1.append(infer_logK_sequences(round_1_sequences[i*len(protein_concentrations) + j], round_1_initial_frequencies[i*len(protein_concentrations) + j], round_1_selected_frequencies[i*len(protein_concentrations) + j], round_1_nonselected_frequencies[i*len(protein_concentrations) + j], error_rates, protein_concentration_1, c, correction_method))
            logK_sequences_r2.append(infer_logK_sequences(round_2_sequences[i*len(protein_concentrations) + j], round_2_initial_frequencies[i*len(protein_concentrations) + j], round_2_selected_frequencies[i*len(protein_concentrations) + j], round_2_nonselected_frequencies[i*len(protein_concentrations) + j], error_rates, protein_concentration_2, c, correction_method))
            mutation_r1, interaction_r1 = infer_logK_mutations(logK_sequences_r1[-1], round_1_sequences[i*len(protein_concentrations) + j], lambda_l1)
            mutation_r2, interaction_r2 = infer_logK_mutations(logK_sequences_r2[-1], round_2_sequences[i*len(protein_concentrations) + j], lambda_l1)
            logK_mutations_r1.append(mutation_r1)
            logK_mutations_r2.append(mutation_r2)
            interactions_r1.append(interaction_r1)
            interactions_r2.append(interaction_r2)

    return logK_sequences_r1, logK_mutations_r1, interactions_r1, logK_sequences_r2, logK_mutations_r2, interactions_r2, single_effects, interctions, round_1_sequence_effects, round_2_sequence_effects

def logK_inference_exp(path : str, protein_concentrations : list, c : float, usable_pools : list, lambda_l1 : float = 0.001, correction_method: str = 'sampling'):

    round_2_sequences, round_2_initial_frequencies, round_2_selected_frequencies, round_2_nonselected_frequencies, error_rates, significant_positions = get_pool_data_exp(path, protein_concentrations)

    logK_sequences_r2 = []
    logK_mutations_r2 = []
    interactions_r2 = []

    savepath = f'/datadisk/MIME/exp/expData/Inference_c_{c}_lambda_{lambda_l1}_{correction_method}/'
    # create directory
    import os
    os.makedirs(savepath, exist_ok=True)

    for i in range(len(protein_concentrations)):
        protein_concentration_1 = protein_concentrations[i]
        for j in range(len(protein_concentrations)):
            protein_concentration_2 = protein_concentrations[j]
            if f'{protein_concentration_1}_{protein_concentration_2}' not in usable_pools:
                continue
            print(f"Pool {protein_concentration_1}, {protein_concentration_2}:")
            logK_sequences_r2.append(infer_logK_sequences(round_2_sequences[i*len(protein_concentrations) + j], round_2_initial_frequencies[i*len(protein_concentrations) + j], round_2_selected_frequencies[i*len(protein_concentrations) + j], round_2_nonselected_frequencies[i*len(protein_concentrations) + j], error_rates, protein_concentration_2, c, correction_method))
            mutation_r2, interaction_r2 = infer_logK_mutations(logK_sequences_r2[-1], round_2_sequences[i*len(protein_concentrations) + j], lambda_l1)
            logK_mutations_r2.append(mutation_r2)
            interactions_r2.append(interaction_r2)
            # save logK sequences, mutations and interactions
            np.savez(f"{savepath}logK_sequences_{protein_concentration_1}_{protein_concentration_2}.npz", logK_sequences=logK_sequences_r2[-1])
            np.savez(f"{savepath}logK_mutations_{protein_concentration_1}_{protein_concentration_2}.npz", logK_mutations=logK_mutations_r2[-1])
            np.savez(f"{savepath}interactions_{protein_concentration_1}_{protein_concentration_2}.npz", interactions=interactions_r2[-1])

    return logK_sequences_r2, logK_mutations_r2, interactions_r2, significant_positions

