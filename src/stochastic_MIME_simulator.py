import numpy as np
from scipy.stats import gmean
from scipy import sparse
from scipy.optimize import minimize
import os


# parameters

# number_sequences = 10000000
# # relative_number_targets = 10
# sequence_length = 20

# number_states = 4
# p_state_change = 2/sequence_length
# p_effect = 0.7

# generate ground truth

def generate_ground_truth(sequence_length : int, number_states : int, p_state_change : float, p_effect) -> np.ndarray:
    # default state has value 1
    default_state = np.ones(sequence_length)
    # mutant states are drawn from a log-normal distribution
    mutant_states = np.round(np.random.lognormal(mean=0, sigma=1, size=(number_states-1, sequence_length)),2)
    # set mutant states to 1 with probability p_effect
    mutant_states[np.random.rand(number_states-1, sequence_length) > p_effect] = 1

    ground_truth = np.row_stack((default_state, mutant_states))
    print(ground_truth)

    return ground_truth

# ground_truth = generate_ground_truth(sequence_length, number_states, p_state_change)
# print(ground_truth)

# generate sequences

def generate_sequences(ground_truth : np.ndarray, number_sequences : int, p_state_change : float, pruning : int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get number_states and sequence length
    number_states, sequence_length = ground_truth.shape

    # generate sequences as random choices over the states
    # sequences = np.random.choice(np.byte(number_states), size=(number_sequences, sequence_length), p=[1-p_state_change] + [p_state_change/(number_states-1)] * (number_states - 1)).astype(np.byte) # byte is enough for 128 states

    # generate number of mutated states
    number_mutated_states = np.random.binomial(sequence_length*number_sequences, p_state_change)
    # generate positions of mutated states
    position_ids = np.random.choice(sequence_length*number_sequences, number_mutated_states, replace=False)
    positions = np.unravel_index(position_ids, (number_sequences, sequence_length))
    # fill sparse matrix with 1s at positions of mutated states
    sequences = sparse.csr_matrix((np.ones(number_mutated_states), positions), shape=(number_sequences, sequence_length), dtype=np.ubyte).toarray()
    # replace 1s with random 8-bit integers
    sequences[sequences == 1] = np.random.randint(1, number_states, sequences[sequences == 1].shape[0], dtype=np.ubyte) # ubyte is enough for 256 states
    # we do this instead of the random choice above so the sequences are generated as 8-bit integers so large arrays can be stored in memory
    print('size full sequence array:', np.round(sequences.nbytes/(1024**2)), 'MB')
    # print(sequences)    

    # condense sequences to unique sequences and add counts as last column
    unique_sequences, counts = np.unique(sequences, axis=0, return_counts=True)
    # print(np.round(unique_sequences.nbytes/(1024**2)), 'MB')
    # print(unique_sequences)
    # print(counts)

    #remove sequences with less than pruning counts
    if pruning > 0:
        unique_sequences = unique_sequences[counts >= pruning]
        counts = counts[counts >= pruning]

    # compute effect of each unique sequence
    # effect of a sequence is the product of the effects of the states per position
    sequence_effects = np.array([np.prod([ground_truth[int(unique_sequences[i,j]), j] for j in range(sequence_length)]) for i in range(unique_sequences.shape[0])])
    # print(sequence_effects)

    return unique_sequences, counts, sequence_effects

def remutate_sequences(unique_sequences : np.ndarray, counts : np.ndarray, ground_truth : np.ndarray, p_state_change : float, pruning : int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # expand unique sequences to full sequences
    sequences = np.repeat(unique_sequences, counts, axis=0)
    # generate number of mutated states
    number_mutated_states = np.random.binomial(sequences.shape[0]*sequences.shape[1], p_state_change)
    # generate positions of mutated states
    position_ids = np.random.choice(sequences.shape[0]*sequences.shape[1], number_mutated_states, replace=False)
    positions = np.unravel_index(position_ids, sequences.shape)
    # overwrite mutated states with random 8-bit integers
    sequences[positions] = np.random.randint(1, ground_truth.shape[0], number_mutated_states, dtype=np.ubyte)
    print('size full sequence array:', np.round(sequences.nbytes/(1024**2)), 'MB')

    # condense sequences to unique sequences and add counts as last column
    new_unique_sequences, new_counts = np.unique(sequences, axis=0, return_counts=True)

    #remove sequences with less than pruning counts
    if pruning > 0:
        new_unique_sequences = new_unique_sequences[new_counts >= pruning]
        new_counts = new_counts[new_counts >= pruning]

    # compute effect of each unique sequence
    # effect of a sequence is the product of the effects of the states per position
    new_sequence_effects = np.array([np.prod([ground_truth[int(new_unique_sequences[i,j]), j] for j in range(sequences.shape[1])]) for i in range(new_unique_sequences.shape[0])])

    return new_unique_sequences, new_counts, new_sequence_effects

def select_pool(counts : np.ndarray, sequence_effects : np.ndarray, relative_number_targets : int) -> tuple[np.ndarray, np.ndarray]:
    # get frequencies from counts
    frequencies = counts / np.sum(counts)
    # get free target concentration by minimizing the objective function
    def objective_function(x):
        return np.square(relative_number_targets - np.sum(frequencies * x / (x + sequence_effects)) - x)
    x0 = np.array([1])
    # define bounds so free target concentration is positive
    bounds = [(0, None)]
    # minimize the objective function
    res = minimize(objective_function, x0, bounds=bounds)
    free_target_concentration = res.x[0]
    print('free target concentration', free_target_concentration)
    # check if the non-squared objective function is close to zero
    print(np.sqrt(objective_function(free_target_concentration)))

    # compute the number of selected sequenes as binomial random variable (stochastic)
    counts_selected = np.random.binomial(counts, (free_target_concentration * counts / (free_target_concentration + sequence_effects))/counts)

    # compute number of non-selected sequences
    counts_non_selected = counts - counts_selected

    return counts_selected, counts_non_selected

def single_site_count(unique_sequences : np.ndarray, counts : np.ndarray, number_states : int, sequence_length : int) -> np.ndarray:
    
    # compute the number of sequences with a given single state at each position
    single_site_counts = np.zeros((number_states, sequence_length))
    for i in range(number_states):
        for j in range(sequence_length):
            # get the row indices of the unique sequences that have state i at position j
            row_indices = np.where(unique_sequences[:,j] == i)[0]
            # sum the counts of these sequences
            single_site_counts[i,j] = np.sum(counts[row_indices])
            if single_site_counts[i,j] == 0: 
                single_site_counts[i,j] = 1 #TODO get a better solution for this
                print('Warning: no sequences with state', i, 'at position', j)
    return single_site_counts

def pairwise_count(unique_sequences : np.ndarray, counts : np.ndarray, number_states : int, sequence_length : int, output_path: str) -> np.ndarray:
    # open text file for writing
    f = open(output_path, 'w')
    #write header
    f.write('pos1\tpos2\t11\t12\t13\t14\t21\t22\t23\t24\t31\t32\t33\t34\t41\t42\t43\t44\n')

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

def check_independence_assumption(ground_truth : np.ndarray, single_site_frequencies: np.array, sequence_effects : np.ndarray, frequencies : np.ndarray) -> np.ndarray:
    # get number_states and sequence length
    number_states, sequence_length = ground_truth.shape

    # compute the geometric mean of the sequence effects
    gmean_sequence_effects = gmean(sequence_effects, weights=frequencies)

    # compute the products of the geometric means of the single site effects (columns are positions, rows are states)
    gmean_single_site_effects = np.prod(gmean(ground_truth, axis=1, weights=single_site_frequencies), axis=0)

    # check if the geometric mean of the sequence effects is equal to the product of the geometric means of the single site effects
    print("gmean sequence effects: ", gmean_sequence_effects)
    print("product of gmean single site effects: ", gmean_single_site_effects)
    print("gmean sequence effects is equal to product of gmean single site effects: ", np.isclose(gmean_sequence_effects, gmean_single_site_effects))

    effects = np.array([gmean_sequence_effects, gmean_single_site_effects])

    return effects

def check_average_assumption(ground_truth : np.ndarray, single_site_effects : np.ndarray, unique_sequences : np.ndarray, sequence_effects : np.ndarray, frequencies : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #compute the true background effect
    true_background = single_site_effects/ground_truth

    # compute the average background effect
    average_background = np.zeros((ground_truth.shape[0], ground_truth.shape[1]))
    for state in range(ground_truth.shape[0]):
        for position in range(ground_truth.shape[1]):
            mutant_indices = np.where(unique_sequences[:,position] == state)[0]
            default_indices = np.where(unique_sequences[:,position] == 0)[0]
            average_background[state, position] = ((gmean(sequence_effects[mutant_indices], weights=frequencies[mutant_indices])/ground_truth[state, position]))/((gmean(sequence_effects[default_indices], weights=frequencies[default_indices])/ground_truth[0, position]))

    # check if the true background effect is equal to the average background effect
    print("true background effect: \n", true_background)
    print("average background effect: \n", average_background)
    print("true background effect is equal to average background effect: ", np.allclose(true_background, average_background))

    return true_background, average_background
    
def simulate_dm_MIME(ground_truth : np.ndarray, number_sequences : int, relative_number_targets_1 : int, relative_number_targets_2 : int,p_state_change : float, output_path : str, pruning : int = 0) -> np.ndarray:
    # get number_states and sequence length
    number_states, sequence_length = ground_truth.shape
    # save ground truth
    os.makedirs(output_path, exist_ok=True)
    np.savetxt(output_path + 'ground_truth.csv', ground_truth[1:].flatten('F'), delimiter=',', fmt='%f')

    # generate sequences
    unique_sequences, counts, sequence_effects = generate_sequences(ground_truth, number_sequences, p_state_change)
    # save unique sequences, counts and sequence effects
    os.makedirs(output_path + 'round_1', exist_ok=True)
    np.savetxt(output_path + 'round_1/unique_sequences.csv', unique_sequences, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_1/counts.csv', counts, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_1/sequence_effects.csv', sequence_effects, delimiter=',', fmt='%f')

    # select sequences
    selected, non_selected = select_pool(counts, sequence_effects, relative_number_targets_1)
    # save selected and non-selected sequences
    os.makedirs(output_path + 'round_1/selected', exist_ok=True)
    os.makedirs(output_path + 'round_1/non_selected', exist_ok=True)
    np.savetxt(output_path + 'round_1/selected/counts.csv', selected, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_1/non_selected/counts.csv', non_selected, delimiter=',', fmt='%d')

    # prune sequences with less than pruning counts
    if pruning > 0:
        pruned_sequences = unique_sequences[counts >= pruning]
        pruned_counts = counts[counts >= pruning]
        pruned_selected = selected[counts >= pruning]
        pruned_non_selected = non_selected[counts >= pruning]
    else:
        pruned_sequences = unique_sequences
        pruned_counts = counts
        pruned_selected = selected
        pruned_non_selected = non_selected

    # compute single site counts
    single_site_counts_selected = single_site_count(pruned_sequences, pruned_selected, number_states, sequence_length)
    single_site_counts_non_selected = single_site_count(pruned_sequences, pruned_non_selected, number_states, sequence_length)
    # save single site counts
    np.savetxt(output_path + 'round_1/selected/single_site_counts.csv', single_site_counts_selected, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_1/non_selected/single_site_counts.csv', single_site_counts_non_selected, delimiter=',', fmt='%d')

    # compute pairwise counts
    pairwise_count(pruned_sequences, pruned_selected, number_states, sequence_length, output_path + 'round_1/selected/pairwise_count.csv')
    pairwise_count(pruned_sequences, pruned_non_selected, number_states, sequence_length, output_path + 'round_1/non_selected/pairwise_count.csv')

    # infer effects
    effects = infer_effects(single_site_counts_selected, single_site_counts_non_selected)
    # save effects
    # remove row 0 (default state), flatten and save
    np.savetxt(output_path + 'round_1/effects.csv', effects[1:].flatten('F'), delimiter=',', fmt='%f')

    # check independence assumption
    single_site_counts = single_site_count(unique_sequences, counts, number_states, sequence_length)
    gmean_effects = check_independence_assumption(ground_truth, single_site_counts, sequence_effects, counts)
    # check average assumption
    true_background, average_background = check_average_assumption(ground_truth, effects, unique_sequences, sequence_effects, counts)
    # create assupmtion directory
    os.makedirs(output_path + 'round_1/assumptions', exist_ok=True)
    # save gmean_effects, true_background and average_background
    np.savetxt(output_path + 'round_1/assumptions/gmean_effects.csv', gmean_effects, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_1/assumptions/true_background.csv', true_background, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_1/assumptions/average_background.csv', average_background, delimiter=',', fmt='%f')

    # enrich non-selected pool to be the same number as the initial pool
    enriched_non_selected_counts = np.round(non_selected * number_sequences / np.sum(non_selected)).astype(int)

    # remutate non-selected sequences
    unique_sequences, counts, sequence_effects = remutate_sequences(unique_sequences, enriched_non_selected_counts, ground_truth, p_state_change)
    os.makedirs(output_path + 'round_2', exist_ok=True)
    np.savetxt(output_path + 'round_2/unique_sequences.csv', unique_sequences, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_2/counts.csv', counts, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_2/sequence_effects.csv', sequence_effects, delimiter=',', fmt='%f')

    # select sequences
    selected, non_selected = select_pool(counts, sequence_effects, relative_number_targets_2)
    # save selected and non-selected sequences
    os.makedirs(output_path + 'round_2/selected', exist_ok=True)
    os.makedirs(output_path + 'round_2/non_selected', exist_ok=True)
    np.savetxt(output_path + 'round_2/selected/counts.csv', selected, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_2/non_selected/counts.csv', non_selected, delimiter=',', fmt='%d')

    # prune sequences with less than pruning counts
    if pruning > 0:
        pruned_sequences = unique_sequences[counts >= pruning]
        pruned_counts = counts[counts >= pruning]
        pruned_selected = selected[counts >= pruning]
        pruned_non_selected = non_selected[counts >= pruning]
    else:
        pruned_sequences = unique_sequences
        pruned_counts = counts
        pruned_selected = selected
        pruned_non_selected = non_selected

    # compute single site counts
    single_site_counts_selected = single_site_count(pruned_sequences, pruned_selected, number_states, sequence_length)
    single_site_counts_non_selected = single_site_count(pruned_sequences, pruned_non_selected, number_states, sequence_length)
    # save single site counts
    np.savetxt(output_path + 'round_2/selected/single_site_counts.csv', single_site_counts_selected, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_2/non_selected/single_site_counts.csv', single_site_counts_non_selected, delimiter=',', fmt='%d')

    # compute pairwise counts
    pairwise_count(pruned_sequences, pruned_selected, number_states, sequence_length, output_path + 'round_2/selected/pairwise_count.csv')
    pairwise_count(pruned_sequences, pruned_non_selected, number_states, sequence_length, output_path + 'round_2/non_selected/pairwise_count.csv')

    # infer effects
    effects = infer_effects(single_site_counts_selected, single_site_counts_non_selected)
    # save effects
    np.savetxt(output_path + 'round_2/effects.csv', effects[1:].flatten('F'), delimiter=',', fmt='%f')

    # check independence assumption
    single_site_counts = single_site_count(unique_sequences, counts, number_states, sequence_length)
    gmean_effects = check_independence_assumption(ground_truth, single_site_counts, sequence_effects, counts)
    # check average assumption
    true_background, average_background = check_average_assumption(ground_truth, effects, unique_sequences, sequence_effects, counts)
    # create assupmtion directory
    os.makedirs(output_path + 'round_2/assumptions', exist_ok=True)
    # save gmean_effects, true_background and average_background
    np.savetxt(output_path + 'round_2/assumptions/gmean_effects.csv', gmean_effects, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_2/assumptions/true_background.csv', true_background, delimiter=',', fmt='%f')
    np.savetxt(output_path + 'round_2/assumptions/average_background.csv', average_background, delimiter=',', fmt='%f')

    print('done')
    return

    


# ground_truth = generate_ground_truth(sequence_length, number_states, p_state_change)
# unique_sequences, counts, sequence_effects = generate_sequences(ground_truth, number_sequences, p_state_change)
# selected, non_selected = select_pool(unique_sequences, counts, sequence_effects, relative_number_targets)
# print(single_site_count(unique_sequences, selected, number_states, sequence_length))
# print(single_site_count(unique_sequences, non_selected, number_states, sequence_length))
# print(np.round(infer_effects(single_site_count(unique_sequences, selected, number_states, sequence_length), single_site_count(unique_sequences, non_selected, number_states, sequence_length)),2))
# pairwise_count(unique_sequences, selected, number_states, sequence_length, 'data/test_data/pairwise_count_selected.csv')
# pairwise_count(unique_sequences, non_selected, number_states, sequence_length, 'data/test_data/pairwise_count_non_selected.csv')
# unique_sequences, counts, sequence_effects = remutate_sequences(unique_sequences, counts, ground_truth, p_state_change)
# selected, non_selected = select_pool(unique_sequences, counts, sequence_effects, relative_number_targets)
# print(single_site_count(unique_sequences, selected, number_states, sequence_length))
# print(single_site_count(unique_sequences, non_selected, number_states, sequence_length))
# print(np.round(infer_effects(single_site_count(unique_sequences, selected, number_states, sequence_length), single_site_count(unique_sequences, non_selected, number_states, sequence_length)),2))


def main(name :str, sequence_length : int = 20, number_states : int = 4, p_state_change : float = 2/20, p_effect : float = 0.7, number_sequences : int = 10000000, pruning : int = 0):
    if sequence_length != 20 and p_state_change == 2/20:
        p_state_change = 2/sequence_length
        
    ground_truth = generate_ground_truth(sequence_length, number_states, p_state_change, p_effect)
    for target1 in [.1, 1, 10]:
        for target2 in [.1, 1, 10]:
            simulate_dm_MIME(ground_truth, number_sequences, target1, target2, p_state_change, f'/datadisk/MIME/{name}/target1_{target1}_target2_{target2}/', pruning)

if __name__ == '__main__':
    main('discrete_L7_q5_n500k_asstest', sequence_length=7, number_sequences=500000, pruning=0)

