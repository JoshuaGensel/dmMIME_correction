import numpy as np
from scipy.optimize import minimize
from scipy.stats import gmean
import matplotlib.pyplot as plt
import os


# # parameters

# relative_number_targets = 1
# sequence_length = 3
# number_states = 4
# p_state_change = 1/sequence_length
# p_effect = .7

# generate ground truth

def generate_ground_truth(sequence_length : int, number_states : int, p_interaction) -> np.ndarray:
    # default state has value e
    default_state = np.ones(sequence_length)  #* np.e #** 2
    # mutant states are drawn from a log-normal distribution
    mutant_states = np.round(np.random.lognormal(mean=0, sigma=1, size=(number_states-1, sequence_length)),2)
    # mutant_states = np.round(np.exp((np.random.beta(a = 4, b = 2, size=(number_states-1, sequence_length))-0)*3),2)
    # set mutant states to 1 with probability 1 - p_effect
    # mutant_states = np.where(np.random.rand(*mutant_states.shape) < 1-p_effect, 1*np.e, mutant_states)

    ground_truth = np.row_stack((default_state, mutant_states))

    print("ground truth: \n", ground_truth)

    # generate interaction matrix
    interaction_matrix = np.zeros((sequence_length*(number_states-1), sequence_length*(number_states-1)))
    interacting_states = np.array([], dtype=int)
        
    for pos in range(sequence_length-1):
        for state in range(number_states-1):
            # check if state is already interacting
            if pos*(number_states-1) + state in interacting_states:
                continue
            # coin toss to determine if there is an interaction with probability p_interaction
            if np.random.rand() < p_interaction:
                # choose a state from the upper triangular part of the matrix to interact with and check if it has already been chosen
                possible_interacting_states = np.arange((pos+1)* (number_states-1), sequence_length*(number_states-1))
                # remove already chosen states
                possible_interacting_states = np.setdiff1d(possible_interacting_states, interacting_states)
                if len(possible_interacting_states) != 0:
                    # choose a state
                    interacting_state = np.random.choice(possible_interacting_states)
                    # add interacting state to array
                    interacting_states = np.append(interacting_states, interacting_state)
                    # set interaction to random normal value
                    interaction_matrix[pos*(number_states-1) + state, interacting_state] = np.round(np.random.normal(-.5, .3),1)

    # make matrix symmetric
    interaction_matrix = interaction_matrix + interaction_matrix.T
    # set diagonal to 1
    np.fill_diagonal(interaction_matrix, 1)

    print("interaction matrix: \n", interaction_matrix)

    return ground_truth, interaction_matrix

# ground_truth = generate_ground_truth(sequence_length, number_states, p_effect)

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

# generate sequences

def generate_sequences(ground_truth : np.ndarray, E : np.ndarray, p_state_change : float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get number_states and sequence length
    number_states, sequence_length = ground_truth.shape

    # create every possible sequence
    sequences = np.array(np.meshgrid(*[np.arange(number_states)]*sequence_length)).T.reshape(-1, sequence_length)

    # calculate the probability of each sequence as p_state_change**number of state changes * (1-p_state_change)**(sequence_length - number of state changes)
    state_changes = np.sum(sequences != np.zeros(sequence_length), axis=1)
    frequencies = (p_state_change/(number_states-1))**state_changes * (1-p_state_change)**(sequence_length - state_changes)

    # # add random normal noise to the frequencies
    # frequencies = frequencies + np.random.normal(0, 0.001, frequencies.shape)
    # # set negative frequencies to 0
    # frequencies = np.where(frequencies < 0, 0, frequencies)
    # # normalize frequencies
    # frequencies = frequencies / np.sum(frequencies)

    # # generate a random frequency for each sequence
    # frequencies = np.random.rand(sequences.shape[0])
    # # normalize frequencies
    # frequencies = frequencies / np.sum(frequencies)

    # # set frequency < 0.001 to 0 with probability p
    # p = 1.5
    # frequencies = np.where(frequencies < 0.01, np.where(np.random.rand(*frequencies.shape) < p, 0, frequencies), frequencies)
    # frequencies = frequencies / np.sum(frequencies)


    # compute effect of each unique sequence
    # transform ground truth into X matrix
    X = np.zeros((sequence_length*(number_states-1), sequence_length*(number_states-1)))
    np.fill_diagonal(X, np.log(ground_truth[1:].flatten("F")))
    # generate A matrix as one-hot encoding of sequences
    A = one_hot_encoding(sequences)
    # sequence effects are rowsums of (AX)*(AE)
    sequence_effects = np.exp(np.sum((A @ X) * (A @ E), axis=1))


    print("\tnumber sequences is ", len(sequences))
    print("\tfrequencies sum to ", np.sum(frequencies))

    return sequences, frequencies, sequence_effects

# sequences, frequencies, sequence_effects = generate_sequences(ground_truth, p_state_change)
# print(sequences)

# print(sequence_effects)

def select_pool(frequencies : np.ndarray, sequence_effects : np.ndarray, relative_number_targets : int) -> tuple[np.ndarray, np.ndarray]:
    
    # get free target concentration by minimizing the objective function
    def objective_function(x):
        return np.square(relative_number_targets - np.sum(frequencies * x / (x + sequence_effects)) - x)
    x0 = np.array([1])
    # define bounds so free target concentration is positive
    bounds = [(0, relative_number_targets)]
    # minimize the objective function
    res = minimize(objective_function, x0, bounds=bounds)
    free_target_concentration = res.x[0]
    print('\tfree target concentration is ', free_target_concentration)
    # check if the non-squared objective function is close to zero
    print("\toptimized value plugged in equation should be 0 and is", np.sqrt(objective_function(free_target_concentration)))

    # compute number of selected sequences
    probability_selected = (free_target_concentration / (free_target_concentration + sequence_effects))

    # compute number of non-selected sequences
    probability_not_selected = 1 - probability_selected

    frequencies_selected = probability_selected * frequencies
    frequencies_not_selected = probability_not_selected * frequencies

    #check if free target concentration is same as initial target concentration - number of selected sequences
    # print("initial target concentration - number of selected sequences is ", relative_number_targets - np.sum(frequencies_selected))

    # normalize frequencies
    # frequencies_selected = frequencies_selected / np.sum(frequencies_selected)
    # frequencies_not_selected = frequencies_not_selected / np.sum(frequencies_not_selected)
    print("\tfrequencies selected sum to ", np.sum(frequencies_selected))
    print("\tfrequencies not selected sum to ", np.sum(frequencies_not_selected))

    return frequencies_selected, frequencies_not_selected

# frequencies_selected, frequencies_not_selected = select_pool(frequencies, sequence_effects, relative_number_targets)

# # barplot of initial frequencies and after selection
# fig, ax = plt.subplots(1, 3, figsize=(15,5))
# ax[0].bar(np.arange(len(frequencies)), frequencies)
# ax[0].set_title("Initial frequencies")
# ax[1].bar(np.arange(len(frequencies_selected)), frequencies_selected)
# ax[1].set_title("Selected frequencies")
# ax[2].bar(np.arange(len(frequencies_not_selected)), frequencies_not_selected)
# ax[2].set_title("Not selected frequencies")
# plt.show()

def remutate_sequences(sequences : np.ndarray, frequencies: np.ndarray, number_states : int, p_state_change : float) -> np.ndarray:
    # get number of sequences and sequence length
    number_sequences, sequence_length = sequences.shape

    # create n by n matrix of necessary mutations to change from each sequence to each other sequence
    mutations = np.zeros((number_sequences, number_sequences))
    for i in range(number_sequences):
        for j in range(i+1, number_sequences):
            mutations[i,j] = np.sum(sequences[i] != sequences[j])
            mutations[j,i] = mutations[i,j]

    # print("mutations matrix: \n", mutations)

    # create a tranistion matrix of probabilities of changing from one sequence to another
    # probability of changing from one sequence to another is p_state_change**number of mutations * (1-p_state_change)**(sequence_length - number of mutations)
    probabilities = (p_state_change/(number_states-1))**mutations * (1-p_state_change)**(sequence_length - mutations)
    # print("probabilities matrix: \n", probabilities)

    # get new frequencies as multiplication of initial frequencies and the transition matrix
    new_frequencies = np.dot(frequencies.T, probabilities)
    # print("new frequencies: \n", new_frequencies)

    # # add random normal noise to the frequencies
    # new_frequencies = new_frequencies + np.random.normal(0, 0.001, new_frequencies.shape)
    # # set negative new_frequencies to 0
    # new_frequencies = np.where(new_frequencies < 0, 0, new_frequencies)
    # # normalize new_frequencies
    # new_frequencies = new_frequencies / np.sum(new_frequencies)


    # # set frequency < 0.001 to 0 with probability 0.5
    # new_frequencies = np.where(new_frequencies < 0.01, np.where(np.random.rand(*new_frequencies.shape) < 1.5, 0, new_frequencies), new_frequencies)
    # new_frequencies = new_frequencies / np.sum(new_frequencies)

    # normalize new frequencies to sum to 1
    # new_frequencies = new_frequencies / np.sum(new_frequencies)
    print("\tfrequencies remutated sum to ", np.sum(new_frequencies))

    return new_frequencies

def add_sequencing_errors(sequences : np.ndarray, frequencies: np.ndarray, number_states : int, p_error : np.ndarray) -> np.ndarray:
    # get number of sequences and sequence length
    number_sequences, sequence_length = sequences.shape

    # normalize frequencies
    # frequencies = frequencies / np.sum(frequencies)

    # p_error is array of probabilities of sequencing errors for each position
    # create a number_sequences by number_sequences matrix of prababilities of changing from one sequence to another
    probabilities = np.zeros((number_sequences, number_sequences))
    for i in range(number_sequences):
        for j in range(number_sequences):
            changes = []
            for position in range(sequence_length):
                # if sequences differ at position, probability of changing from one sequence to another is p_error
                if sequences[i,position] != sequences[j,position]:
                    changes.append(p_error[position]/(number_states-1))
                else:
                    changes.append(1-p_error[position])
            probabilities[i,j] = np.prod(changes)

    # print("\tprobabilities matrix: \n", np.round(probabilities, 2))

    

    # get error frequencies as multiplication of initial frequencies and the transition matrix
    new_frequencies = np.dot(frequencies.T, probabilities)

    # check if new frequencies sum to 1
    print("\tfrequencies with errors sum to ", np.sum(new_frequencies))
    # check if frequencies are identical to frequencies without errors
    # print("frequencies with errors are equal to frequencies without errors: ", np.allclose(new_frequencies, frequencies))

    return new_frequencies

# new_frequencies = remutate_sequences(sequences, frequencies, number_states, p_state_change)

# # plot initial and new frequencies
# fig, ax = plt.subplots(1, 2, figsize=(10,5))
# ax[0].bar(np.arange(len(frequencies)), frequencies)
# ax[0].set_title("Initial frequencies")
# ax[1].bar(np.arange(len(new_frequencies)), new_frequencies)
# ax[1].set_title("New frequencies")
# plt.show()

def single_site_frequency(sequences : np.ndarray, frequencies : np.ndarray, number_states : int, sequence_length : int) -> np.ndarray:

    # create a matrix of frequencies of each state at each position
    state_frequencies = np.zeros((number_states, sequence_length))
    for i in range(number_states):
        for j in range(sequence_length):
            # get the row indices of the unique sequences that have state i at position j
            row_indices = np.where(sequences[:,j] == i)[0]
            # sum the counts of these sequences
            state_frequencies[i,j] = np.sum(frequencies[row_indices])

    return state_frequencies

# state_frequencies = single_site_frequency(sequences, frequencies, number_states, sequence_length)
# print("state frequencies: \n", state_frequencies)
# state_frequencies_selected = single_site_frequency(sequences, frequencies_selected, number_states, sequence_length)
# print("state frequencies selected: \n", state_frequencies_selected)
# state_frequencies_not_selected = single_site_frequency(sequences, frequencies_not_selected, number_states, sequence_length)
# print("state frequencies not selected: \n", state_frequencies_not_selected)
# state_frequencies_new = single_site_frequency(sequences, new_frequencies, number_states, sequence_length)
# print("state frequencies new: \n", state_frequencies_new)

def pairwise_frequency(sequences : np.ndarray, frequencies : np.ndarray, number_states : int, sequence_length : int) -> np.ndarray:

    # create a matrix of frequencies of each pair of states at each position
    pair_frequencies = np.zeros((number_states * sequence_length, number_states * sequence_length))
    for position1 in range(sequence_length):
        for position2 in range(position1+1, sequence_length):
            for i in range(number_states):
                for j in range(number_states):
                    # get the row indices of the unique sequences that have state i at position1 and state j at position2
                    row_indices = np.where((sequences[:,position1] == i) & (sequences[:,position2] == j))[0]
                    # sum the counts of these sequences
                    pair_frequencies[position1*number_states + i, position2*number_states + j] = np.sum(frequencies[row_indices])

    # set diagonal to 1
    np.fill_diagonal(pair_frequencies, 1)

    # make matrix symmetric
    pair_frequencies = (pair_frequencies + pair_frequencies.T - np.diag(pair_frequencies.diagonal()))
    return pair_frequencies

# pair_frequencies = pairwise_frequency(sequences, frequencies, number_states, sequence_length)
# print("pair frequencies: \n", np.round(pair_frequencies,3))

def pairwise_count(unique_sequences : np.ndarray, counts : np.ndarray, number_states : int, sequence_length : int, output_path: str) -> np.ndarray:
    # open text file for writing
    f = open(output_path, 'w')
    #write header
    f.write('pos1\tpos2')
    for state1 in range(number_states):
        for state2 in range(number_states):
            f.write('\t' + str(state1) + str(state2))
    f.write('\n')

    for pos1 in range(sequence_length):
        for pos2 in range(pos1+1, sequence_length):
            f.write(str(pos1+1) + '\t' + str(pos2+1) + '\t')
            # compute the number of sequences with a given pair of states at each position
            for state1 in range(number_states):
                for state2 in range(number_states):
                    # get the row indices of the unique sequences that have state i at position j
                    row_indices = np.where((unique_sequences[:,pos1] == state1) & (unique_sequences[:,pos2] == state2))[0]
                    # sum the counts of these sequences
                    count = np.sum(counts[row_indices])
                    f.write(str(count) + '\t')
            # delete last tab
            f.seek(f.tell()-1)
            # write newline
            f.write('\n')
    f.close()
    return

def infer_effects(single_site_counts_selected : np.ndarray, single_site_counts_non_selected : np.ndarray) -> np.ndarray:
    # first divide both matrices by their first row (default state) elementwise
    single_site_counts_selected = single_site_counts_selected / single_site_counts_selected[0]
    single_site_counts_non_selected = single_site_counts_non_selected / single_site_counts_non_selected[0]

    # then divide the non-selected matrix by the selected matrix elementwise
    return single_site_counts_non_selected / single_site_counts_selected

# single_site_counts_selected = single_site_frequency(sequences, frequencies_selected, number_states, sequence_length)
# single_site_counts_not_selected = single_site_frequency(sequences, frequencies_not_selected, number_states, sequence_length)
# effects = infer_effects(single_site_counts_selected, single_site_counts_not_selected)
# print("effects: \n", effects)

def check_independence_assumption(ground_truth : np.ndarray, single_site_frequencies: np.array, sequence_effects : np.ndarray, frequencies : np.ndarray) -> np.ndarray:
    # get number_states and sequence length
    number_states, sequence_length = ground_truth.shape

    # compute the geometric mean of the sequence effects
    gmean_sequence_effects = gmean(sequence_effects, weights=frequencies)

    # compute the products of the geometric means of the single site effects (columns are positions, rows are states)
    gmean_single_site_effects = np.prod(gmean(ground_truth, axis=1, weights=single_site_frequencies), axis=0)

    # check if the geometric mean of the sequence effects is equal to the product of the geometric means of the single site effects
    # print("gmean sequence effects: ", gmean_sequence_effects)
    # print("product of gmean single site effects: ", gmean_single_site_effects)
    # print("gmean sequence effects is equal to product of gmean single site effects: ", np.isclose(gmean_sequence_effects, gmean_single_site_effects))

    effects = np.array([gmean_sequence_effects, gmean_single_site_effects])

    return effects

def check_average_assumption(ground_truth : np.ndarray, single_site_effects : np.ndarray, unique_sequences : np.ndarray, sequence_effects : np.ndarray, frequencies : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #compute the true background effect
    true_background = single_site_effects/(ground_truth/ground_truth[0])

    # compute the average background effect
    average_background = np.zeros((ground_truth.shape[0], ground_truth.shape[1]))
    gmean_ratio = np.zeros((ground_truth.shape[0], ground_truth.shape[1]))
    for state in range(ground_truth.shape[0]):
        for position in range(ground_truth.shape[1]):
            mutant_indices = np.where(unique_sequences[:,position] == state)[0]
            default_indices = np.where(unique_sequences[:,position] == 0)[0]
            average_background[state, position] = ((gmean(sequence_effects[mutant_indices], weights=frequencies[mutant_indices])/ground_truth[state, position]))/((gmean(sequence_effects[default_indices], weights=frequencies[default_indices])/ground_truth[0, position]))
            gmean_ratio[state, position] = gmean(sequence_effects[mutant_indices], weights=frequencies[mutant_indices])/gmean(sequence_effects[default_indices], weights=frequencies[default_indices])

    # check if the true background effect is equal to the average background effect
    # print("true background effect: \n", true_background)
    # print("average background effect: \n", average_background)
    # print("true background effect is equal to average background effect: ", np.allclose(true_background, average_background))

    return true_background, average_background, gmean_ratio


def simulate_dm_MIME(ground_truth : np.ndarray, interaction_matrix : np.ndarray, relative_number_targets_1 : int, relative_number_targets_2 : int,p_state_change : float, p_error : np.ndarray, output_path : str) -> np.ndarray:
    # get number_states and sequence length
    number_states, sequence_length = ground_truth.shape
    # save ground truth
    os.makedirs(output_path, exist_ok=True)
    np.savetxt(output_path + 'ground_truth.csv', ground_truth[1:].flatten('F'), delimiter=',', fmt='%f')
    np.savetxt(output_path + 'interaction_matrix.csv', interaction_matrix, delimiter=',', fmt='%f')

    # generate sequences
    unique_sequences, counts, sequence_effects = generate_sequences(ground_truth, interaction_matrix, p_state_change)
    # save unique sequences, counts and sequence effects
    os.makedirs(output_path + 'round_1', exist_ok=True)
    np.savetxt(output_path + 'round_1/unique_sequences.csv', unique_sequences, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_1/counts.csv', counts, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_1/sequence_effects.csv', sequence_effects, delimiter=',', fmt='%f')

    # select sequences
    selected, non_selected = select_pool(counts, sequence_effects, relative_number_targets_1)
    # save selected and non-selected sequences
    os.makedirs(output_path + 'round_1/selected', exist_ok=True)
    os.makedirs(output_path + 'round_1/non_selected', exist_ok=True)
    np.savetxt(output_path + 'round_1/selected/counts.csv', selected, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_1/non_selected/counts.csv', non_selected, delimiter=',', fmt='%f')

    # add sequencing errors
    selected_counts_with_error = add_sequencing_errors(unique_sequences, selected, number_states, p_error)
    non_selected_counts_with_error = add_sequencing_errors(unique_sequences, non_selected, number_states, p_error)
    # save selected and non-selected sequences
    np.savetxt(output_path + 'round_1/selected/counts_with_error.csv', selected_counts_with_error, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_1/non_selected/counts_with_error.csv', non_selected_counts_with_error, delimiter=',', fmt='%f')

    # compute single site counts
    single_site_counts_selected = single_site_frequency(unique_sequences, selected, number_states, sequence_length)
    single_site_counts_non_selected = single_site_frequency(unique_sequences, non_selected, number_states, sequence_length)
    # save single site counts
    np.savetxt(output_path + 'round_1/selected/single_site_counts.csv', single_site_counts_selected, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_1/non_selected/single_site_counts.csv', single_site_counts_non_selected, delimiter=',', fmt='%f')

    # compute pairwise counts
    pairwise_count(unique_sequences, selected, number_states, sequence_length, output_path + 'round_1/selected/pairwise_count.csv')
    pairwise_count(unique_sequences, non_selected, number_states, sequence_length, output_path + 'round_1/non_selected/pairwise_count.csv')

    # infer effects
    effects = infer_effects(single_site_counts_selected, single_site_counts_non_selected)
    # save effects
    # remove row 0 (default state), flatten and save
    np.savetxt(output_path + 'round_1/effects.csv', effects[1:].flatten('F'), delimiter=',', fmt='%f')

    # check independence assumption
    single_site_counts = single_site_frequency(unique_sequences, counts, number_states, sequence_length)
    gmean_effects = check_independence_assumption(ground_truth, single_site_counts, sequence_effects, counts)
    # check average assumption
    true_background, average_background, gmean_ratio = check_average_assumption(ground_truth, effects, unique_sequences, sequence_effects, counts)
    # create assupmtion directory
    os.makedirs(output_path + 'round_1/assumptions', exist_ok=True)
    # save gmean_effects, true_background and average_background
    np.savetxt(output_path + 'round_1/assumptions/gmean_effects.csv', gmean_effects, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_1/assumptions/true_background.csv', true_background, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_1/assumptions/average_background.csv', average_background, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_1/assumptions/gmean_ratio.csv', gmean_ratio, delimiter=',', fmt='%f')

    # normalize non-selected counts
    non_selected = non_selected / np.sum(non_selected)

    # remutate non-selected sequences
    counts = remutate_sequences(unique_sequences, non_selected, number_states, p_state_change)
    os.makedirs(output_path + 'round_2', exist_ok=True)
    np.savetxt(output_path + 'round_2/counts.csv', counts, delimiter=',', fmt='%f')
    # save unique sequences and sequence effects again (they are the same as in round 1)
    np.savetxt(output_path + 'round_2/unique_sequences.csv', unique_sequences, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_2/sequence_effects.csv', sequence_effects, delimiter=',', fmt='%f')

    # select sequences
    selected, non_selected = select_pool(counts, sequence_effects, relative_number_targets_2)
    # save selected and non-selected sequences
    os.makedirs(output_path + 'round_2/selected', exist_ok=True)
    os.makedirs(output_path + 'round_2/non_selected', exist_ok=True)
    np.savetxt(output_path + 'round_2/selected/counts.csv', selected, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_2/non_selected/counts.csv', non_selected, delimiter=',', fmt='%f')

    # add sequencing errors
    selected_counts_with_error = add_sequencing_errors(unique_sequences, selected, number_states, p_error)
    non_selected_counts_with_error = add_sequencing_errors(unique_sequences, non_selected, number_states, p_error)
    # save selected and non-selected sequences
    np.savetxt(output_path + 'round_2/selected/counts_with_error.csv', selected_counts_with_error, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_2/non_selected/counts_with_error.csv', non_selected_counts_with_error, delimiter=',', fmt='%f')


    # compute single site counts
    single_site_counts_selected = single_site_frequency(unique_sequences, selected, number_states, sequence_length)
    single_site_counts_non_selected = single_site_frequency(unique_sequences, non_selected, number_states, sequence_length)
    # save single site counts
    np.savetxt(output_path + 'round_2/selected/single_site_counts.csv', single_site_counts_selected, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_2/non_selected/single_site_counts.csv', single_site_counts_non_selected, delimiter=',', fmt='%f')

    # compute pairwise counts
    pairwise_count(unique_sequences, selected, number_states, sequence_length, output_path + 'round_2/selected/pairwise_count.csv')
    pairwise_count(unique_sequences, non_selected, number_states, sequence_length, output_path + 'round_2/non_selected/pairwise_count.csv')

    # infer effects
    effects = infer_effects(single_site_counts_selected, single_site_counts_non_selected)
    # save effects
    np.savetxt(output_path + 'round_2/effects.csv', effects[1:].flatten('F'), delimiter=',', fmt='%f')

    # check independence assumption
    single_site_counts = single_site_frequency(unique_sequences, counts, number_states, sequence_length)
    gmean_effects = check_independence_assumption(ground_truth, single_site_counts, sequence_effects, counts)
    # check average assumption
    true_background, average_background, gmean_ratio = check_average_assumption(ground_truth, effects, unique_sequences, sequence_effects, counts)
    # create assupmtion directory
    os.makedirs(output_path + 'round_2/assumptions', exist_ok=True)
    # save gmean_effects, true_background and average_background
    np.savetxt(output_path + 'round_2/assumptions/gmean_effects.csv', gmean_effects, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_2/assumptions/true_background.csv', true_background, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_2/assumptions/average_background.csv', average_background, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_2/assumptions/gmean_ratio.csv', gmean_ratio, delimiter=',', fmt='%f')


    print('done')
    return

def main(name :str, sequence_length : int = 5, number_states : int = 4, p_state_change : float = 2/5, p_error : float = 2/(5*2), p_interaction : float = 0.5):
    if sequence_length != 5 and p_state_change == 2/5:
        p_state_change = 2/sequence_length
    if p_state_change != 2/5 and p_error == 2/(5*5):
        p_error = p_state_change/2
    print(f'simulating MIME for {name}')

    # generate error rates
    mean_error = np.log(p_error)-(1/20)
    error_rates = np.random.lognormal(mean_error, .1, sequence_length)
    print('error rates: ', error_rates)

    # generate wildtype error count matrix
    error_counts = np.zeros((number_states, sequence_length))
    for i in range(sequence_length):
        error_counts[1:,i] = error_rates[i]/(number_states-1)
        error_counts[0,i] = 1 - error_rates[i]

    # write error rates and counts to file
    np.savetxt(f'/datadisk/MIME/{name}/error_rates.csv', error_rates, delimiter=',', fmt='%f')
    np.savetxt(f'/datadisk/MIME/{name}/error_counts.csv', error_counts, delimiter=',', fmt='%f')

    ground_truth, interaction_matrix = generate_ground_truth(sequence_length, number_states, p_interaction)
    for target1 in [.1, 1, 10]:
        for target2 in [.1, 1, 10]:
            print(f'simulating MIME for target1 {target1} and target2 {target2}')
            simulate_dm_MIME(ground_truth, interaction_matrix, target1, target2, p_state_change, error_rates, f'/datadisk/MIME/{name}/target1_{target1}_target2_{target2}/')

    #write parameters to file
    f = open(f'/datadisk/MIME/{name}/parameters.txt', 'w')
    f.write(f'sequence_length: {sequence_length}\n')
    f.write(f'number_states: {number_states}\n')
    f.write(f'p_state_change: {p_state_change}\n')
    f.write(f'p_interaction: {p_interaction}\n')
    f.close()

    print('finished')

    return

if __name__ == '__main__':
    main('deterministic_error05_L5', sequence_length=10, number_states=4, p_state_change=1/5, p_error=1/10, p_interaction=0.5)
