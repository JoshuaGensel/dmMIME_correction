import numpy as np
from scipy.optimize import minimize
import os


# parameters

relative_number_targets = 1
sequence_length = 3
number_states = 2
p_state_change = 1/sequence_length
p_effect = .7
# generate ground truth

def generate_ground_truth(sequence_length : int, number_states : int, p_effect) -> np.ndarray:
    # default state has value 1
    default_state = np.ones(sequence_length)
    # mutant states are drawn from a log-normal distribution
    mutant_states = np.round(np.random.lognormal(mean=0, sigma=1, size=(number_states-1, sequence_length)),2)
    # set mutant states to 1 with probability 1 - p_effect
    mutant_states = np.where(np.random.rand(*mutant_states.shape) < 1-p_effect, 1, mutant_states)

    ground_truth = np.row_stack((default_state, mutant_states))

    return ground_truth

ground_truth = generate_ground_truth(sequence_length, number_states, p_effect)
print("ground truth: \n", ground_truth)

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


    return sequences, frequencies, sequence_effects

sequences, frequencies, sequence_effects = generate_sequences(ground_truth, p_state_change)
# print(sequences)
print("number sequences is ", len(sequences))
print("frequencies sum to ", np.sum(frequencies))
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
    print('free target concentration', free_target_concentration)
    # check if the non-squared objective function is close to zero
    print("optimized value plugged in equation should be 0 and is", np.sqrt(objective_function(free_target_concentration)))

    # compute number of selected sequences
    frequencies_selected = free_target_concentration * frequencies / (free_target_concentration + sequence_effects)

    # compute number of non-selected sequences
    frequencies_not_selected = frequencies - frequencies_selected

    # normalize frequencies
    frequencies_selected = frequencies_selected / np.sum(frequencies_selected)
    frequencies_not_selected = frequencies_not_selected / np.sum(frequencies_not_selected)

    return frequencies_selected, frequencies_not_selected

frequencies_selected, frequencies_not_selected = select_pool(frequencies, sequence_effects, relative_number_targets)

print("frequencies selected sum to ", np.sum(frequencies_selected))
print("frequencies not selected sum to ", np.sum(frequencies_not_selected))

# barplot of initial frequencies and after selection
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15,5))
ax[0].bar(np.arange(len(frequencies)), frequencies)
ax[0].set_title("Initial frequencies")
ax[1].bar(np.arange(len(frequencies_selected)), frequencies_selected)
ax[1].set_title("Selected frequencies")
ax[2].bar(np.arange(len(frequencies_not_selected)), frequencies_not_selected)
ax[2].set_title("Not selected frequencies")
plt.show()