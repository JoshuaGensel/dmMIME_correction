import numpy as np
from scipy.stats import gmean
from scipy import sparse


# parameters

number_targets = 5
number_sequences = 10
sequence_length = 3
number_states = 4
p_state_change = 0.2

# generate ground truth

def generate_ground_truth(sequence_length : int, number_states : int, p_state_change : float) -> np.ndarray:
    # default state has value 1
    default_state = np.ones(sequence_length)
    # mutant states are drawn from a log-normal distribution
    mutant_states = np.round(np.random.lognormal(mean=0, sigma=1, size=(number_states-1, sequence_length)),2)

    ground_truth = np.row_stack((default_state, mutant_states))
    # print(ground_truth)

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
    sequences = sparse.csr_matrix((np.ones(number_mutated_states), positions), shape=(number_sequences, sequence_length), dtype=np.byte).toarray()
    # replace 1s with random 8-bit integers
    sequences[sequences == 1] = np.random.randint(1, number_states, sequences[sequences == 1].shape[0], dtype=np.byte)
    # we do this instead of the random choice above so the sequences are generated as 8-bit integers so large arrays can be stored in memory
    print(np.round(sequences.nbytes/(1024**2)), 'MB')
    # print(sequences)    

    # condense sequences to unique sequences and add counts as last column
    unique_sequences, counts = np.unique(sequences, axis=0, return_counts=True)
    sequences_with_counts = np.column_stack((unique_sequences, counts))
    print(np.round(sequences_with_counts.nbytes/(1024**2)), 'MB')
    print(sequences_with_counts)

    # compute effect of each unique sequence
    # effect of a sequence is the product of the effects of the states per position
    sequence_effects = np.array([np.prod([ground_truth[int(sequences_with_counts[i,j]), j] for j in range(sequence_length)]) for i in range(unique_sequences.shape[0])])
    print(sequence_effects)
    # print(np.column_stack((sequences_with_counts, effects)).astype(float))

    #TODO: see if you need to return sequences (largest array) or if you can just return the condensed version for remutation
    return sequences, sequences_with_counts, sequence_effects


generate_sequences(generate_ground_truth(sequence_length, number_states, p_state_change), number_sequences, p_state_change)
print('done')
    