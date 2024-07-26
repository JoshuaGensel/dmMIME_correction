import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os


# # parameters

# relative_number_targets = 1
# sequence_length = 3
# number_states = 4
# p_state_change = 1/sequence_length
# p_effect = .7

# generate ground truth

def generate_ground_truth(sequence_length : int, number_states : int, p_effect) -> np.ndarray:
    # default state has value 1
    default_state = np.ones(sequence_length)
    # mutant states are drawn from a log-normal distribution
    mutant_states = np.round(np.random.lognormal(mean=0, sigma=1, size=(number_states-1, sequence_length)),2)
    # set mutant states to 1 with probability 1 - p_effect
    mutant_states = np.where(np.random.rand(*mutant_states.shape) < 1-p_effect, 1, mutant_states)

    ground_truth = np.row_stack((default_state, mutant_states))

    print("ground truth: \n", ground_truth)

    return ground_truth

# ground_truth = generate_ground_truth(sequence_length, number_states, p_effect)

# generate sequences

def generate_sequences(ground_truth : np.ndarray, p_state_change : float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get number_states and sequence length
    number_states, sequence_length = ground_truth.shape

    # create every possible sequence
    sequences = np.array(np.meshgrid(*[np.arange(number_states)]*sequence_length)).T.reshape(-1, sequence_length)

    # calculate the probability of each sequence as p_state_change**number of state changes * (1-p_state_change)**(sequence_length - number of state changes)
    state_changes = np.sum(sequences != np.zeros(sequence_length), axis=1)
    frequencies = (p_state_change/(number_states-1))**state_changes * (1-p_state_change)**(sequence_length - state_changes)

    # compute effect of each unique sequence
    # effect of a sequence is the product of the effects of the states per position
    sequence_effects = np.array([np.prod([ground_truth[int(sequences[i,j]), j] for j in range(sequence_length)]) for i in range(sequences.shape[0])])

    print("number sequences is ", len(sequences))
    print("frequencies sum to ", np.sum(frequencies))

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
    print('free target concentration is ', free_target_concentration)
    # check if the non-squared objective function is close to zero
    print("optimized value plugged in equation should be 0 and is", np.sqrt(objective_function(free_target_concentration)))

    # compute number of selected sequences
    frequencies_selected = free_target_concentration * frequencies / (free_target_concentration + sequence_effects)

    # compute number of non-selected sequences
    frequencies_not_selected = frequencies - frequencies_selected

    # normalize frequencies
    frequencies_selected = frequencies_selected / np.sum(frequencies_selected)
    frequencies_not_selected = frequencies_not_selected / np.sum(frequencies_not_selected)
    print("frequencies selected sum to ", np.sum(frequencies_selected))
    print("frequencies not selected sum to ", np.sum(frequencies_not_selected))

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

    # create a matrix of probabilities of changing from one sequence to another
    # probability of changing from one sequence to another is frequency of the first sequence * p_state_change**number of mutations * (1-p_state_change)**(sequence_length - number of mutations)
    probabilities = frequencies[:, None] * (p_state_change/(number_states-1))**mutations * (1-p_state_change)**(sequence_length - mutations)
    # print("probabilities matrix: \n", probabilities)

    # summing over the rows gives the new frequencies
    new_frequencies = np.sum(probabilities, axis=0)
    # print("new frequencies: \n", new_frequencies)
    print("frequencies remutated sum to ", np.sum(new_frequencies))

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


def simulate_dm_MIME(ground_truth : np.ndarray, relative_number_targets_1 : int, relative_number_targets_2 : int,p_state_change : float, output_path : str) -> np.ndarray:
    # get number_states and sequence length
    number_states, sequence_length = ground_truth.shape
    # save ground truth
    os.makedirs(output_path, exist_ok=True)
    np.savetxt(output_path + 'ground_truth.csv', ground_truth[1:].flatten('F'), delimiter=',', fmt='%f')

    # generate sequences
    unique_sequences, counts, sequence_effects = generate_sequences(ground_truth, p_state_change)
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

    # remutate non-selected sequences
    counts = remutate_sequences(unique_sequences, non_selected, number_states, p_state_change)
    os.makedirs(output_path + 'round_2', exist_ok=True)
    np.savetxt(output_path + 'round_2/counts.csv', counts, delimiter=',', fmt='%f')

    # select sequences
    selected, non_selected = select_pool(counts, sequence_effects, relative_number_targets_2)
    # save selected and non-selected sequences
    os.makedirs(output_path + 'round_2/selected', exist_ok=True)
    os.makedirs(output_path + 'round_2/non_selected', exist_ok=True)
    np.savetxt(output_path + 'round_2/selected/counts.csv', selected, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_2/non_selected/counts.csv', non_selected, delimiter=',', fmt='%f')


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

    print('done')
    return

def main(name :str, sequence_length : int = 20, number_states : int = 4, p_state_change : float = 2/20, p_effect : float = 0.7):
    if sequence_length != 20 and p_state_change == 2/20:
        p_state_change = 2/sequence_length
        
    ground_truth = generate_ground_truth(sequence_length, number_states, p_effect)
    for target1 in [.1, 1, 10]:
        for target2 in [.1, 1, 10]:
            simulate_dm_MIME(ground_truth, target1, target2, p_state_change, f'/datadisk/MIME/{name}/target1_{target1}_target2_{target2}/')

if __name__ == '__main__':
    main('deterministic_L_5_q_4', sequence_length=5, number_states=4, p_state_change=1/5, p_effect=0.7)
