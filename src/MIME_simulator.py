import numpy as np
from scipy.stats import gmean
from scipy import sparse
from scipy.optimize import minimize


# parameters

number_targets = 1000000
number_sequences = 2000000
sequence_length = 5
number_states = 4
p_state_change = 0.2

# generate ground truth

def generate_ground_truth(sequence_length : int, number_states : int, p_state_change : float) -> np.ndarray:
    # default state has value 1
    default_state = np.ones(sequence_length)
    # mutant states are drawn from a log-normal distribution
    mutant_states = np.round(np.random.lognormal(mean=0, sigma=1, size=(number_states-1, sequence_length)),2)

    ground_truth = np.row_stack((default_state, mutant_states))
    print(ground_truth)

    return ground_truth

# ground_truth = generate_ground_truth(sequence_length, number_states, p_state_change)
# print(ground_truth)

# generate sequences

def generate_sequences(ground_truth : np.ndarray, number_sequences : int, p_state_change : float) -> np.ndarray:
    # get number_sattes and sequence length
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

def select_pool(unique_sequences : np.ndarray, counts : np.ndarray, sequence_effects : np.ndarray, number_targets : int) -> tuple[np.ndarray, np.ndarray]:
    # get free target concentration by minimizing the objective function
    def objective_function(x):
        return np.square(number_targets - np.sum((x*counts)/(x + sequence_effects)) - x)
    x0 = np.array([1])
    res = minimize(objective_function, x0, method='Nelder-Mead')
    free_target_concentration = res.x[0]
    print('free target concentration', free_target_concentration)
    # check if the non-squared objective function is close to zero
    # print(objective_function(free_target_concentration))

    # compute number of selected sequences
    counts_selected = np.round(free_target_concentration * counts / (free_target_concentration + sequence_effects)).astype(int)

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

def infer_effects(single_site_counts_selected : np.ndarray, single_site_counts_non_selected : np.ndarray) -> np.ndarray:
    # first divide both matrices by their first row (default state) elementwise
    single_site_counts_selected = single_site_counts_selected / single_site_counts_selected[0]
    single_site_counts_non_selected = single_site_counts_non_selected / single_site_counts_non_selected[0]

    # then divide the non-selected matrix by the selected matrix elementwise
    return single_site_counts_non_selected / single_site_counts_selected
    


unique_sequences, counts, sequence_effects = generate_sequences(generate_ground_truth(sequence_length, number_states, p_state_change), number_sequences, p_state_change)
selected, non_selected = select_pool(unique_sequences, counts, sequence_effects, number_targets)
# print(single_site_count(unique_sequences, selected, number_states, sequence_length))
# print(single_site_count(unique_sequences, non_selected, number_states, sequence_length))
print(np.round(infer_effects(single_site_count(unique_sequences, selected, number_states, sequence_length), single_site_count(unique_sequences, non_selected, number_states, sequence_length)),2))
print('done')
    