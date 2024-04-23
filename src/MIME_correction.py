import numpy as np
import sympy as sym
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def get_inferred_Kds(path : str):
    """
    Get inferred Kds from PositionWiseKdEstimates.csv file
    """
    df = pd.read_csv(path, sep='\t')
    median_mut_C = df['median mut C'].values
    median_mut_G = df['median mut G'].values
    median_mut_U = df['median mut U'].values
    median_Kds = np.vstack((median_mut_C, median_mut_G, median_mut_U)).T.flatten()
    return median_Kds
    

def construct_frequency_matrix(path_to_pairwise_counts_unbound : str, path_to_pairwise_counts_bound : str):
    """Constructs the frequency matrix from the pairwise count files output by the dmMIME simulator.
    First the bound and unbound counts are added together to get the total counts of the initial pool.
    Then a square n x n matrix is constructed where n is the number of possible mutations, so sequence length * 3.
    For mutations at the same position the entry is 0, which are 3x3 blocks along the diagonal.
    For the mutation of interest the entry is 1, which is 1 along the diagonal.
    For mutations at different positions the entry is the pairwise frequency of both mutations - the pairwise frequency of the mutation and the wildtype.


    Args:
        path_to_pairwise_counts_unbound (str): path to the pairwise count file for unbound sequences output by the dmMIME simulator (8.txt)
        path_to_pairwise_counts_bound (str): path to the pairwise count file for bound sequences output by the dmMIME simulator (7.txt)
    """

    unbound_counts = np.loadtxt(path_to_pairwise_counts_unbound, skiprows=1, delimiter='\t')
    bound_counts = np.loadtxt(path_to_pairwise_counts_bound, skiprows=1, delimiter='\t')

    # first 2 columns for positions stay the same, the rest are the counts
    counts = np.hstack((unbound_counts[:, :2], unbound_counts[:, 2:] + bound_counts[:, 2:]))

    # get single counts
    single_counts_unbound = np.loadtxt(path_to_pairwise_counts_unbound.replace('2d', '1d'), skiprows=1, delimiter='\t')
    single_counts_bound = np.loadtxt(path_to_pairwise_counts_bound.replace('2d', '1d'), skiprows=1, delimiter='\t')

    single_counts = single_counts_unbound[:, 1:] + single_counts_bound[:, 1:]
    single_freqs = single_counts/single_counts.sum(axis=1)[:, None]


    # construct the frequency matrix
    n_pos = np.where(counts[:, 0] == 1)[0].shape[0] + 1
    n_mut =  n_pos * 3
    freq_matrix = np.zeros((n_mut, n_mut))

    for pos1 in range(n_pos):
        for pos2 in range(n_pos):
            if pos1 == pos2:
                for mut1 in range(3):
                    for mut2 in range(3):
                    # if mutations are at the same position
                        if mut1 == mut2:
                            freq_matrix[pos1*3 + mut1, pos2*3 + mut2] = 1
                        else:
                            freq_matrix[pos1*3 + mut1, pos2*3 + mut2] = 0

            else:
                for mut1 in range(3):
                    for mut2 in range(3):
                        # TODO this part still needs correction of the counts for sequencing errors? I think this cancels out because you multiply the counts above and below fraction by the same number
                        if pos1 < pos2:
                            pairwise_counts = counts[np.where((counts[:, 0] == pos1 + 1) & (counts[:, 1] == pos2 + 1))[0], 2:]
                            # get pairwise frequency of the mutation 2 at position 2 and the wildtype at position 1
                            freq_mut2_wt1 = pairwise_counts[0, mut2+1]/(pairwise_counts[0, 0] + pairwise_counts[0, 1] + pairwise_counts[0, 2] + pairwise_counts[0, 3])
                            # get pairwise frequency of the mutation 2 at position 2 and the mutation 1 at position 1
                            freq_mut2_mut1 = pairwise_counts[0, (mut1+1)*4 + mut2+1]/(pairwise_counts[0, (mut1+1)*4] + pairwise_counts[0, (mut1+1)*4 + 1] + pairwise_counts[0, (mut1+1)*4 + 2] + pairwise_counts[0, (mut1+1)*4 + 3])

                            freq_matrix[pos1*3 + mut1, pos2*3 + mut2] = freq_mut2_mut1 - freq_mut2_wt1

                        if pos1 > pos2:
                            # TODO: this is probably wrong, make sure this works 100%
                            pairwise_counts = counts[np.where((counts[:, 0] == pos2 + 1) & (counts[:, 1] == pos1 + 1))[0], 2:]
                            # get pairwise frequency of the mutation 2 at position 2 and the wildtype at position 1
                            freq_mut2_wt1 = pairwise_counts[0, (mut2+1)*4]/(pairwise_counts[0, 0] + pairwise_counts[0, 4] + pairwise_counts[0, 8] + pairwise_counts[0, 12])
                            # get pairwise frequency of the mutation 2 at position 2 and the mutation 1 at position 1
                            freq_mut2_mut1 = pairwise_counts[0, (mut1+1) + (mut2+1)*4]/(pairwise_counts[0, (mut1+1)+0] + pairwise_counts[0, (mut1+1)+4] + pairwise_counts[0, (mut1+1)+8] + pairwise_counts[0, (mut1+1)+12])

                            freq_matrix[pos1*3 + mut1, pos2*3 + mut2] = freq_mut2_mut1 - freq_mut2_wt1

    print("frequency matrix")
    print(freq_matrix.shape)
    # check if the matrix is full rank
    if np.linalg.matrix_rank(freq_matrix) < n_mut:
        print('matrix is not full rank')
        print('rank of matrix: ', np.linalg.matrix_rank(freq_matrix))
        print('condition number of matrix: ', np.linalg.cond(freq_matrix))
    else:
        print('matrix is full rank')
        #print('rank of matrix: ', np.linalg.matrix_rank(freq_matrix))
        print('condition number of matrix: ', np.linalg.cond(freq_matrix))
    print(np.round(freq_matrix[:9, :9], 2))

    return freq_matrix

def correct_Kds(path_to_pool_data : str):
    """
    This function corrects the Kd estimated made by MIMEAn2 on dmMIME simulated data. It uses 
    pairwise counts to correct for the expected background that each mutation has. For this a 
    frequency matrix containing the cooccurance frequencies of every mutation and wildtype is 
    constructed. Then the true effects can be inferred from solving an equation system of the form
    Ax= b, where A is the frequency matrix, x are the true effects and b are the observed effects.



    Args:
        path_to_pool_data (str): path to the pool directory as constructed from the dmMIME simulator and MIMEAn2 automation scripts. The directory has to contain /2d/7.txt, /2d/8.txt, /result/PositionWiseKdEstimates.csv and /single_kds.txt
    """

    # get inferred Kds
    inferred_Kds = get_inferred_Kds(path_to_pool_data + '/results/PositionWiseKdEstimates.csv')
    
    # get ground truth Kds
    ground_truth_Kds = np.loadtxt(path_to_pool_data + '/single_kds.txt')

    #print number of nans in estimated Kds
    print('number of nans in inferred Kds: ', np.sum(np.isnan(inferred_Kds)))
    # replace nans in inferred Kds with ground truth Kds
    # TODO resolve this: either use MIMEAn2 without quality control or delete mutations from the frequency matrix
    inferred_Kds[np.isnan(inferred_Kds)] = ground_truth_Kds[np.isnan(inferred_Kds)]
    inferred_Kds = np.log(inferred_Kds)
    ground_truth_Kds = np.log(ground_truth_Kds)

    # construct frequency matrix
    # check if 7.txt and 8.txt exist
    if os.path.isfile(path_to_pool_data + '/2d/7.txt') and os.path.isfile(path_to_pool_data + '/2d/8.txt'):
        freq_matrix = construct_frequency_matrix(path_to_pool_data + '/2d/8.txt', path_to_pool_data + '/2d/7.txt')
    else:
        freq_matrix = construct_frequency_matrix(path_to_pool_data + '/2d/4.txt', path_to_pool_data + '/2d/3.txt')

    # solve the equation system
    corrected_Kds = np.linalg.solve(freq_matrix, inferred_Kds)

    return np.exp(corrected_Kds), np.exp(inferred_Kds), np.exp(ground_truth_Kds)



def comparison_plot(ground_truth_Kds, inferred_Kds_1, inferred_Kds_2, inferred_Kds_3, corrected_Kds_1, corrected_Kds_2, corrected_Kds_3, title : str):
    """
    Plot the comparison of ground truth vs inferred Kds and corrected Kds
    """
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # # plot ground truth vs inferred median Kds of each protein concentration
    # ax1.scatter(ground_truth_Kds, inferred_Kds_1, label='prot 0.1', alpha=0.5)
    # ax1.scatter(ground_truth_Kds, inferred_Kds_2, label='prot 1', alpha=0.5)
    # ax1.scatter(ground_truth_Kds, inferred_Kds_3, label='prot 10', alpha=0.5)
    # ax1.set_xlabel('ground truth Kd')
    # ax1.set_ylabel('inferred median Kd')
    # ax1.legend()
    # x = np.linspace(0, np.max(ground_truth_Kds), 100)
    # y = x
    # ax1.plot(x, y, color='black', linestyle='--')

    # # plot log ground truth vs log inferred median Kds of each protein concentration
    # ax2.scatter(np.log(ground_truth_Kds), np.log(inferred_Kds_1), label='prot 0.1', alpha=0.5)
    # ax2.scatter(np.log(ground_truth_Kds), np.log(inferred_Kds_2), label='prot 1', alpha=0.5)
    # ax2.scatter(np.log(ground_truth_Kds), np.log(inferred_Kds_3), label='prot 10', alpha=0.5)
    # ax2.set_xlabel('log ground truth Kd')
    # ax2.set_ylabel('log inferred median Kd')
    # ax2.legend()
    # x = np.linspace(np.min(np.log(ground_truth_Kds)), np.max(np.log(ground_truth_Kds)), 100)
    # y = x
    # ax2.plot(x, y, color='black', linestyle='--')

    # # set overall title
    # fig.suptitle(title)

    # plt.show()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

    # # plot ground truth vs corrected median Kds of each protein concentration
    # ax1.scatter(ground_truth_Kds, corrected_Kds_1, label='prot 0.1', alpha=0.5)
    # ax1.scatter(ground_truth_Kds, corrected_Kds_2, label='prot 1', alpha=0.5)
    # ax1.scatter(ground_truth_Kds, corrected_Kds_3, label='prot 10', alpha=0.5)
    # ax1.set_xlabel('ground truth Kd')
    # ax1.set_ylabel('corrected median Kd')
    # ax1.legend()
    # x = np.linspace(0, np.max(ground_truth_Kds), 100)
    # y = x
    # ax1.plot(x, y, color='black', linestyle='--')

    # # plot log ground truth vs log corrected median Kds of each protein concentration
    # ax2.scatter(np.log(ground_truth_Kds), np.log(corrected_Kds_1), label='prot 0.1', alpha=0.5)
    # ax2.scatter(np.log(ground_truth_Kds), np.log(corrected_Kds_2), label='prot 1', alpha=0.5)
    # ax2.scatter(np.log(ground_truth_Kds), np.log(corrected_Kds_3), label='prot 10', alpha=0.5)
    # ax2.set_xlabel('log ground truth Kd')
    # ax2.set_ylabel('log corrected median Kd')
    # ax2.legend()
    # x = np.linspace(np.min(np.log(ground_truth_Kds)), np.max(np.log(ground_truth_Kds)), 100)
    # y = x
    # ax2.plot(x, y, color='black', linestyle='--')

    # # set overall title
    # fig.suptitle(title + ', corrected')

    # plt.show()
    
    # print squared error for inferred Kds where inferred Kds are not nan
    print('squared error for inferred Kds')
    print(np.round(np.nanmean(np.square(ground_truth_Kds - inferred_Kds_1)), 3))
    print(np.round(np.nanmean(np.square(ground_truth_Kds - inferred_Kds_2)), 3))
    print(np.round(np.nanmean(np.square(ground_truth_Kds - inferred_Kds_3)), 3))

    # print squared error for corrected Kds where corrected Kds are not nan
    print('squared error for corrected Kds')
    print(np.round(np.nanmean(np.square(ground_truth_Kds - corrected_Kds_1)), 3))
    print(np.round(np.nanmean(np.square(ground_truth_Kds - corrected_Kds_2)), 3))
    print(np.round(np.nanmean(np.square(ground_truth_Kds - corrected_Kds_3)), 3))

    # print the difference between the mean of the inferred Kds and the mean of the corrected Kds
    print('difference between squared error for inferred Kds and corrected Kds')
    print(np.round(np.nanmean(np.square(ground_truth_Kds - inferred_Kds_1)) - np.nanmean(np.square(ground_truth_Kds - corrected_Kds_1)), 3))
    print(np.round(np.nanmean(np.square(ground_truth_Kds - inferred_Kds_2)) - np.nanmean(np.square(ground_truth_Kds - corrected_Kds_2)), 3))
    print(np.round(np.nanmean(np.square(ground_truth_Kds - inferred_Kds_3)) - np.nanmean(np.square(ground_truth_Kds - corrected_Kds_3)), 3))

    # # plot the mean of the inferred Kds and corrected Kds
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # # plot ground truth vs inferred median Kds of each protein concentration
    # ax1.scatter(ground_truth_Kds, np.mean([inferred_Kds_1, inferred_Kds_2, inferred_Kds_3], axis=0), label='inferred', alpha=0.5)
    # ax1.scatter(ground_truth_Kds, np.mean([corrected_Kds_1, corrected_Kds_2, corrected_Kds_3], axis=0), label='corrected', alpha=0.5)
    # ax1.set_xlabel('ground truth Kd')
    # ax1.set_ylabel('mean Kd')
    # ax1.legend()
    # x = np.linspace(0, np.max(ground_truth_Kds), 100)
    # y = x
    # ax1.plot(x, y, color='black', linestyle='--')

    # # plot log ground truth vs log inferred median Kds of each protein concentration
    # ax2.scatter(np.log(ground_truth_Kds), np.log(np.mean([inferred_Kds_1, inferred_Kds_2, inferred_Kds_3], axis=0)), label='inferred', alpha=0.5)
    # ax2.scatter(np.log(ground_truth_Kds), np.log(np.mean([corrected_Kds_1, corrected_Kds_2, corrected_Kds_3], axis=0)), label='corrected', alpha=0.5)
    # ax2.set_xlabel('log ground truth Kd')
    # ax2.set_ylabel('log mean Kd')
    # ax2.legend()
    # x = np.linspace(np.min(np.log(ground_truth_Kds)), np.max(np.log(ground_truth_Kds)), 100)
    # y = x
    # ax2.plot(x, y, color='black', linestyle='--')

    # # set overall title
    # fig.suptitle(title + ', mean')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # plot log ground truth vs log inferred median Kds of each protein concentration
    ax1.scatter(np.log(ground_truth_Kds), np.log(inferred_Kds_1), label='prot 0.1', alpha=0.5)
    ax1.scatter(np.log(ground_truth_Kds), np.log(inferred_Kds_2), label='prot 1', alpha=0.5)
    ax1.scatter(np.log(ground_truth_Kds), np.log(inferred_Kds_3), label='prot 10', alpha=0.5)
    ax1.set_xlabel('log ground truth Kd')
    ax1.set_ylabel('log inferred median Kd')
    ax1.legend()
    x = np.linspace(np.min(np.log(ground_truth_Kds)), np.max(np.log(ground_truth_Kds)), 100)
    y = x
    ax1.plot(x, y, color='black', linestyle='--')

    # plot log ground truth vs log corrected median Kds of each protein concentration
    ax2.scatter(np.log(ground_truth_Kds), np.log(corrected_Kds_1), label='prot 0.1', alpha=0.5)
    ax2.scatter(np.log(ground_truth_Kds), np.log(corrected_Kds_2), label='prot 1', alpha=0.5)
    ax2.scatter(np.log(ground_truth_Kds), np.log(corrected_Kds_3), label='prot 10', alpha=0.5)
    ax2.set_xlabel('log ground truth Kd')
    ax2.set_ylabel('log corrected median Kd')
    ax2.legend()
    x = np.linspace(np.min(np.log(ground_truth_Kds)), np.max(np.log(ground_truth_Kds)), 100)
    y = x
    ax2.plot(x, y, color='black', linestyle='--')

    # plot log ground truth vs log inferred median Kds of each protein concentration
    ax3.scatter(np.log(ground_truth_Kds), np.log(np.mean([inferred_Kds_1, inferred_Kds_2, inferred_Kds_3], axis=0)), label='inferred', alpha=0.5)
    ax3.scatter(np.log(ground_truth_Kds), np.log(np.mean([corrected_Kds_1, corrected_Kds_2, corrected_Kds_3], axis=0)), label='corrected', alpha=0.5)
    ax3.set_xlabel('log ground truth Kd')
    ax3.set_ylabel('log mean Kd')
    ax3.legend()
    x = np.linspace(np.min(np.log(ground_truth_Kds)), np.max(np.log(ground_truth_Kds)), 100)
    y = x
    ax3.plot(x, y, color='black', linestyle='--')

    # set overall title
    fig.suptitle(title + ', log')


    plt.show()

    # print the squared error for the mean of the inferred Kds and the mean of the corrected Kds
    print('squared error for mean of inferred Kds and corrected Kds')
    print(np.round(np.nanmean(np.square(ground_truth_Kds - np.mean([inferred_Kds_1, inferred_Kds_2, inferred_Kds_3], axis=0))), 3))
    print(np.round(np.nanmean(np.square(ground_truth_Kds - np.mean([corrected_Kds_1, corrected_Kds_2, corrected_Kds_3], axis=0))), 3))

    # print the difference between the mean of the inferred Kds and the mean of the corrected Kds
    print('difference between squared error for mean of inferred Kds and corrected Kds')
    print(np.round(np.nanmean(np.square(ground_truth_Kds - np.mean([inferred_Kds_1, inferred_Kds_2, inferred_Kds_3], axis=0))) - np.nanmean(np.square(ground_truth_Kds - np.mean([corrected_Kds_1, corrected_Kds_2, corrected_Kds_3], axis=0))),3))

    
          
    
