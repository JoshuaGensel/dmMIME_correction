import numpy as np
from scipy.stats import gmean
from scipy import sparse
from scipy.optimize import minimize
import os


# parameters

number_sequences = 10000000
# relative_number_targets = 10
sequence_length = 20

number_states = 4
p_state_change = 2/sequence_length
p_effect = 0.7

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

def generate_sequences(ground_truth : np.ndarray, number_sequences : int, p_state_change : float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # compute effect of each unique sequence
    # effect of a sequence is the product of the effects of the states per position
    sequence_effects = np.array([np.prod([ground_truth[int(unique_sequences[i,j]), j] for j in range(sequence_length)]) for i in range(unique_sequences.shape[0])])
    # print(sequence_effects)

    return unique_sequences, counts, sequence_effects

def remutate_sequences(unique_sequences : np.ndarray, counts : np.ndarray, ground_truth : np.ndarray, p_state_change : float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

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

    # compute effect of each unique sequence
    # effect of a sequence is the product of the effects of the states per position
    new_sequence_effects = np.array([np.prod([ground_truth[int(new_unique_sequences[i,j]), j] for j in range(sequences.shape[1])]) for i in range(new_unique_sequences.shape[0])])

    return new_unique_sequences, new_counts, new_sequence_effects

def select_pool(unique_sequences : np.ndarray, counts : np.ndarray, sequence_effects : np.ndarray, relative_number_targets : int) -> tuple[np.ndarray, np.ndarray]:
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

    # compute number of selected sequences (deterministic rounding to integers )
    # counts_selected = np.round(free_target_concentration * counts / (free_target_concentration + sequence_effects)).astype(int)
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
    return single_site_counts.astype(int)

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
    
def simulate_dm_MIME(ground_truth : np.ndarray, number_sequences : int, relative_number_targets_1 : int, relative_number_targets_2 : int,p_state_change : float, output_path : str) -> np.ndarray:
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
    selected, non_selected = select_pool(unique_sequences, counts, sequence_effects, relative_number_targets_1)
    # save selected and non-selected sequences
    os.makedirs(output_path + 'round_1/selected', exist_ok=True)
    os.makedirs(output_path + 'round_1/non_selected', exist_ok=True)
    np.savetxt(output_path + 'round_1/selected/counts.csv', selected, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_1/non_selected/counts.csv', non_selected, delimiter=',', fmt='%d')

    # compute single site counts
    single_site_counts_selected = single_site_count(unique_sequences, selected, number_states, sequence_length)
    single_site_counts_non_selected = single_site_count(unique_sequences, non_selected, number_states, sequence_length)
    # save single site counts
    np.savetxt(output_path + 'round_1/selected/single_site_counts.csv', single_site_counts_selected, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_1/non_selected/single_site_counts.csv', single_site_counts_non_selected, delimiter=',', fmt='%d')

    # compute pairwise counts
    pairwise_count(unique_sequences, selected, number_states, sequence_length, output_path + 'round_1/selected/pairwise_count.csv')
    pairwise_count(unique_sequences, non_selected, number_states, sequence_length, output_path + 'round_1/non_selected/pairwise_count.csv')

    # infer effects
    effects = infer_effects(single_site_counts_selected, single_site_counts_non_selected)
    # save effects
    # remove row 0 (default state), flatten and save
    np.savetxt(output_path + 'round_1/effects.csv', effects[1:].flatten('F'), delimiter=',', fmt='%f')

    # enrich non-selected pool to be the same number as the initial pool
    enriched_non_selected_counts = np.round(non_selected * number_sequences / np.sum(non_selected)).astype(int)

    # remutate non-selected sequences
    unique_sequences, counts, sequence_effects = remutate_sequences(unique_sequences, enriched_non_selected_counts, ground_truth, p_state_change)
    os.makedirs(output_path + 'round_2', exist_ok=True)
    np.savetxt(output_path + 'round_2/unique_sequences.csv', unique_sequences, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_2/counts.csv', counts, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_2/sequence_effects.csv', sequence_effects, delimiter=',', fmt='%f')

    # select sequences
    selected, non_selected = select_pool(unique_sequences, counts, sequence_effects, relative_number_targets_2)
    # save selected and non-selected sequences
    os.makedirs(output_path + 'round_2/selected', exist_ok=True)
    os.makedirs(output_path + 'round_2/non_selected', exist_ok=True)
    np.savetxt(output_path + 'round_2/selected/counts.csv', selected, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_2/non_selected/counts.csv', non_selected, delimiter=',', fmt='%d')

    # compute single site counts
    single_site_counts_selected = single_site_count(unique_sequences, selected, number_states, sequence_length)
    single_site_counts_non_selected = single_site_count(unique_sequences, non_selected, number_states, sequence_length)
    # save single site counts
    np.savetxt(output_path + 'round_2/selected/single_site_counts.csv', single_site_counts_selected, delimiter=',', fmt='%d')
    np.savetxt(output_path + 'round_2/non_selected/single_site_counts.csv', single_site_counts_non_selected, delimiter=',', fmt='%d')

    # compute pairwise counts
    pairwise_count(unique_sequences, selected, number_states, sequence_length, output_path + 'round_2/selected/pairwise_count.csv')
    pairwise_count(unique_sequences, non_selected, number_states, sequence_length, output_path + 'round_2/non_selected/pairwise_count.csv')

    # infer effects
    effects = infer_effects(single_site_counts_selected, single_site_counts_non_selected)
    # save effects
    np.savetxt(output_path + 'round_2/effects.csv', effects[1:].flatten('F'), delimiter=',', fmt='%f')

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


def main():
    ground_truth = generate_ground_truth(sequence_length, number_states, p_state_change, p_effect)
    for target1 in [.1, 1, 10]:
        for target2 in [.1, 1, 10]:
            simulate_dm_MIME(ground_truth, number_sequences, target1, target2, p_state_change, '/datadisk/MIME/depth_test/target1_' + str(target1) + '_target2_' + str(target2) + '/')

if __name__ == '__main__':
    main()

