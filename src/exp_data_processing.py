import re
import shutil
import numpy as np
from tqdm import tqdm
import os 
import random 


# parse cigar string into binary string where 1 indicates a match and 0 indicates a mismatch
def parse_cigar(cigar_string):
    # get only numbers from cigar string
    numbers = re.findall(r'\d+', cigar_string)
    # get only letters from cigar string
    letters = re.findall(r'[A-Z]', cigar_string)
    # iterate through numbers
    sequence = ''
    for i in range(len(numbers)):
        # if letter is not a valid symbol, set number to 0
        if letters[i] == 'M':
            sequence += '1' * int(numbers[i])
        else:
            sequence += '0' * int(numbers[i])
    return sequence

# parse quality string into binary string where 1 indicates a high quality base and 0 indicates a low quality base
def parse_quality(quality_string):
    valid_symbols = ('?', "@", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I')
    sequence = ''
    for i in range(len(quality_string)):
        if quality_string[i] in valid_symbols:
            sequence += '1'
        else:
            sequence += '0'
    return sequence

# combine cigar and quality strings into one binary string
def parse_cigar_quality(cigar_string, quality_string):
    quality_seq = parse_quality(quality_string)
    cigar_seq = parse_cigar(cigar_string)
    sequence = ''
    for i in range(len(cigar_seq)):
        if cigar_seq[i] == '1' and quality_seq[i] == '1':
            sequence += '1'
        else:
            sequence += '0'
    return sequence

def parse_sequence(sequence, cigar_string, quality_string):
    valid_symbols = ('A', 'C', 'G', 'T')
    checked_seq = parse_cigar_quality(cigar_string, quality_string)
    parsed_sequence = ''
    for i in range(len(checked_seq)):
        if checked_seq[i] == '1' and sequence[i] in valid_symbols:
            parsed_sequence += sequence[i]
        else:
            parsed_sequence += '0'
    return parsed_sequence

def align_fragment(length_sequence, pos, cigar_string, sequence, quality_string):
    parsed_sequence = parse_sequence(sequence, cigar_string, quality_string)
    # get only numbers from cigar string
    numbers = re.findall(r'\d+', cigar_string)
    # get only letters from cigar string
    letters = re.findall(r'[A-Z]', cigar_string)
    offset = 0
    # iterate through letters till find M
    for i in range(len(letters)):
        if letters[i] != 'M':
            offset += int(numbers[i])
        else:
            break
    aligned_sequence = ''
    # add pos - 1 0s to the beginning of the sequence
    aligned_sequence += '0' * (pos - 1 - offset)
    # add parsed sequence to the end of the sequence
    aligned_sequence += parsed_sequence
    if (pos - 1 - offset) < 0:
        # remove extra 0s from the beginning of the sequence
        aligned_sequence = aligned_sequence[-(pos - 1 - offset):]
    # add 0s to the end of the sequence
    aligned_sequence += '0' * (length_sequence - len(aligned_sequence))
    return aligned_sequence

# merge aligned sequences
def merge_sequences(left_sequence, right_sequence, length_sequence):
    merged_sequence = ''
    for i in range(length_sequence):
        if left_sequence[i] == right_sequence[i]:
            merged_sequence += left_sequence[i]
        # if left sequence is 0, add right sequence
        elif left_sequence[i] == '0':
            merged_sequence += right_sequence[i]
        # if right sequence is 0, add left sequence
        elif right_sequence[i] == '0':
            merged_sequence += left_sequence[i]
        # if left and right sequences are different, add 0
        else:
            merged_sequence += '0'

    return merged_sequence

def align_reads_experimental(file_path_reads_left: str, file_path_reads_right: str, file_path_output: str, sequence_length: int):

    # read in the left and right reads line by line
    with open(file_path_reads_left, 'r') as reads_file_left:
        reads_left = reads_file_left.readlines()
        # remove the new line character at the end of each line
        reads_left = [read.strip() for read in reads_left]

    with open(file_path_reads_right, 'r') as reads_file_right:
        reads_right = reads_file_right.readlines()
        # remove the new line character at the end of each line
        reads_right = [read.strip() for read in reads_right]

    # remove first 3 lines from both left and right reads
    reads_left = reads_left[3:]
    reads_right = reads_right[3:]

    # separate lines at tab character
    reads_left = [read.split('\t') for read in reads_left]
    reads_right = [read.split('\t') for read in reads_right]

    # check if column 1 of left and right reads match
    for i in range(len(reads_left)):
        if reads_left[i][0] != reads_right[i][0]:
            print('Error: reads are not in the same order')
            return None
        
    # only use columns 2,3,4,5,6,10 and 11
    reads_left = [[read[1], read[2], read[3], read[4], read[5], read[9], read[10]] for read in reads_left]
    reads_right = [[read[1], read[2], read[3], read[4], read[5], read[9], read[10]] for read in reads_right]

    # print(reads_left[0])
    # print(reads_right[0])

    # convert to numpy array of strings
    reads_left = np.array(reads_left, dtype='str')
    reads_right = np.array(reads_right, dtype='str')

    # only keep reads with flag value 0 or 16 in left and flag value 0 or 16 in right
    indices_left = np.where((reads_left[:,0] == '0') | (reads_left[:,0] == '16'))
    indices_right = np.where((reads_right[:,0] == '0') | (reads_right[:,0] == '16'))
    indices = np.intersect1d(indices_left, indices_right)
    reads_left = reads_left[indices]
    reads_right = reads_right[indices]

    # print('Number of reads after flag filtering: ', len(reads_left))

    # only keep reads with MAPQ value >= 70 in left and right
    indices_left = np.where(reads_left[:,3].astype('int') >= 70)
    indices_right = np.where(reads_right[:,3].astype('int') >= 70)
    indices = np.intersect1d(indices_left, indices_right)
    reads_left = reads_left[indices]
    reads_right = reads_right[indices]

    # print('Number of reads after MAPQ filtering: ', len(reads_left))

    # only keep reads with CIGAR strings that do not contain 'I' or 'D' in left and right
    indices = np.where((np.char.find(reads_left[:,4], 'I') == -1) & (np.char.find(reads_left[:,4], 'D') == -1) & (np.char.find(reads_right[:,4], 'I') == -1) & (np.char.find(reads_right[:,4], 'D') == -1))
    reads_left = reads_left[indices]
    reads_right = reads_right[indices]

    # print('Number of reads after Insertion/Deletion filtering: ', len(reads_left))

    # align left and right reads
    # then merge and write to file
    with open(file_path_output, 'w') as output_file:
        for i in range(len(reads_left)):
            # align left and right reads
            aligned_sequence_left = align_fragment(sequence_length, reads_left[i][2].astype(int), reads_left[i][4], reads_left[i][5], reads_left[i][6])
            aligned_sequence_right = align_fragment(sequence_length, reads_right[i][2].astype(int), reads_right[i][4], reads_right[i][5], reads_right[i][6])
            # merge aligned sequences
            merged_sequence = merge_sequences(aligned_sequence_left, aligned_sequence_right, sequence_length)
            # write to file
            output_file.write(merged_sequence + '\n')

    return None




# align_reads_experimental('./data/test_data/experimental/left.1.sam', './data/test_data/experimental/right.2.sam', './data/test_data/experimental/aligned_reads.txt', 535)

def remove_non_significant_positions(file_path_input: str, file_path_output: str, significant_positions : list, position_threshold: int ):
    """
    Remove non-significant positions from the aligned reads file. Then if a read has less than position_threshold positions identified (e.g. not '0'), remove the read from the file.
    """
    # read in the aligned reads file line by line
    with open(file_path_input, 'r') as aligned_reads_file:
        aligned_reads = aligned_reads_file.readlines()
        # remove the new line character at the end of each line
        aligned_reads = [read.strip() for read in aligned_reads]
    # remove non-significant positions from the aligned reads
    filtered_reads = []
    for read in aligned_reads:
        filtered_read = ''
        for i in range(len(read)):
            if i+1 in significant_positions:
                filtered_read += read[i]
            else:
                continue
        filtered_reads.append(filtered_read)

    # replace every '0' with 'N'
    filtered_reads = [read.replace('0', 'N') for read in filtered_reads]
    # remove reads with less than position_threshold positions identified
    filtered_reads = [read for read in filtered_reads if len(re.findall(r'[^N]', read)) >= position_threshold]
    # write the filtered reads to the output file
    with open(file_path_output, 'w') as output_file:
        for read in filtered_reads:
            output_file.write(read + '\n')
    return None

# remove_non_significant_positions('./data/test_data/experimental/aligned_reads.txt', './data/test_data/experimental/aligned_reads_filtered.txt', [0, 1, 2, 3, 50,51,52, 500,501,502, 534],2)

def encode_mutations(file_path_input: str, file_path_output: str, reference_sequence: str, significant_positions: list):
    """
    Encode mutations in the aligned reads file. The mutations are encoded as 1 for a mutation and 0 for no mutation. 
    The reference sequence is used to identify the mutations.
    """
    # read in the reference sequence from second line of the fasta file
    with open(reference_sequence, 'r') as reference_file:
        reference_sequence = reference_file.readlines()
        # remove the new line character at the end of each line
        reference_sequence = [line.strip() for line in reference_sequence]
    # remove the first line of the fasta file
    reference_sequence = reference_sequence[1]

    # only keep the significant positions in the reference sequence
    filtered_reference_sequence = ''
    for i in range(len(reference_sequence)):
        if i in significant_positions:
            filtered_reference_sequence += reference_sequence[i]
        else:
            continue
    reference_sequence = filtered_reference_sequence           

    # read in the aligned reads file
    with open(file_path_input, 'r') as aligned_reads_file:
        aligned_reads = aligned_reads_file.readlines()
        # remove the new line character at the end of each line
        aligned_reads = [read.strip() for read in aligned_reads]
    
    order_mutations = ['A', 'C', 'G', 'T']
    # encode wildtype as 0 and mutations as 1, 2 and 3 according to the order of the mutations
    for i in range(len(aligned_reads)):
        # get the mutations in the read
        nucleotides = [aligned_reads[i][j] for j in range(len(aligned_reads[i]))]
        for position in range(len(nucleotides)):
            wildtype = reference_sequence[position]
            if nucleotides[position] == wildtype or nucleotides[position] == 'N':
                nucleotides[position] = '0'
            else:
                # temporarily remove the wildtype from the order mutations list
                order_mutations.remove(wildtype)
                # get the index of the mutation in the order mutations list
                mutation_index = order_mutations.index(nucleotides[position])
                # encode the mutation as 1, 2 or 3
                nucleotides[position] = str(mutation_index + 1)
                # add the wildtype back to the order mutations list at the correct position
                order_mutations.insert(mutation_index, wildtype)
        # replace the nucleotides in the read with the encoded mutations
        aligned_reads[i] = ''.join(nucleotides)
    # write the encoded reads to the output file
    with open(file_path_output, 'w') as output_file:
        for read in aligned_reads:
            output_file.write(read + '\n')
    return None

    


# encode_mutations('./data/test_data/experimental/aligned_reads_filtered.txt', './data/test_data/experimental/aligned_reads_encoded.txt', './data/test_data/experimental/5NL43.fasta', [0, 1, 2, 3, 50,51,52, 500,501,502, 534])

def count_sequences(path_to_bound_encoded_seqs : str, path_to_unbound_encoded_seqs : str, path_to_output : str):
    """
    Gets each unique sequence from the bound and unbound encoded sequences and counts the number of times each sequence appears as bound and unbound.
    Then writes the unique sequences, the boudn and the unbound counts to separate files.
    """

    # read in the bound and unbound encoded sequences
    with open(path_to_bound_encoded_seqs, 'r') as bound_file:
        bound_sequences = bound_file.readlines()
        # remove the new line character at the end of each line
        bound_sequences = [read.strip() for read in bound_sequences]

    with open(path_to_unbound_encoded_seqs, 'r') as unbound_file:
        unbound_sequences = unbound_file.readlines()
        # remove the new line character at the end of each line
        unbound_sequences = [read.strip() for read in unbound_sequences]
    
    # get unique sequences
    unique_sequences = set(bound_sequences + unbound_sequences)
    
    # count the number of times each sequence appears in the bound and unbound sequences
    bound_counts = []
    unbound_counts = []
    for sequence in tqdm(unique_sequences):
        bound_counts.append(bound_sequences.count(sequence))
        unbound_counts.append(unbound_sequences.count(sequence))
    
    # make output directory if it doesn't exist
    os.makedirs(path_to_output + '/encoded_pool/', exist_ok=True)

    # write the unique sequences, the boudn and the unbound counts to separate files
    with open(path_to_output + '/encoded_pool/unique_sequences.txt', 'w') as output_file:
        for sequence in unique_sequences:
            output_file.write(sequence + '\n')
    
    with open(path_to_output + '/encoded_pool/bound_counts.txt', 'w') as output_file:
        for count in bound_counts:
            output_file.write(str(count) + '\n')
    
    with open(path_to_output + '/encoded_pool/unbound_counts.txt', 'w') as output_file:
        for count in unbound_counts:
            output_file.write(str(count) + '\n')

    return None

# count_sequences('./data/test_data/experimental/aligned_reads_encoded_bound.txt', './data/test_data/experimental/aligned_reads_encoded_unbound.txt', './data/test_data/experimental')

# align_reads_experimental('/datadisk/MIME/exp/expData/GAG_UB_8.1.sam', '/datadisk/MIME/exp/expData/GAG_UB_8.2.sam', './data/8_unbound_aligned_reads.txt', 535)

# get significant position list
significant_positions = np.loadtxt('/datadisk/MIME/exp/expData/sig_pos.txt', dtype='int').tolist()
# print(f'Number of significant positions: {len(significant_positions)}')
# print(significant_positions)

# remove_non_significant_positions('./data/8_unbound_aligned_reads.txt', './data/8_unbound_aligned_reads_filtered.txt', significant_positions, 2)
# remove_non_significant_positions('./data/8_bound_aligned_reads.txt', './data/8_bound_aligned_reads_filtered.txt', significant_positions, 2)

# encode_mutations('./data/8_unbound_aligned_reads_filtered.txt', './data/8_unbound_aligned_reads_encoded.txt', '/datadisk/MIME/exp/expData/5NL43.fasta', significant_positions)
# encode_mutations('./data/8_bound_aligned_reads_filtered.txt', './data/8_bound_aligned_reads_encoded.txt', '/datadisk/MIME/exp/expData/5NL43.fasta', significant_positions)

# count_sequences('./data/8_bound_aligned_reads_encoded.txt', './data/8_unbound_aligned_reads_encoded.txt', './data/8_counts')

def automated_parsing(data_directory : str, protein_concentrations : list, 
                     position_threshold : int, path_to_reference_sequence : str, 
                     significant_positions_file : str, output_directory : str):
    """
    Automates the parsing of experimental data. 
    """
    # make output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # get significant position list
    significant_positions = np.loadtxt(significant_positions_file, dtype='int').tolist()

    # get sequence length from reference sequence
    with open(path_to_reference_sequence, 'r') as reference_file:
        reference_sequence = reference_file.readlines()
        # remove the new line character at the end of each line
        reference_sequence = [line.strip() for line in reference_sequence]
    # remove the first line of the fasta file
    reference_sequence = reference_sequence[1]
    sequence_length = len(reference_sequence)
    print(f'Sequence length: {sequence_length}')
    print(f'Number of significant positions: {len(significant_positions)}')
    
    # first round:
    for concentration in protein_concentrations:
        print(f'Processing concentration: {concentration}')
        print('\tAligning reads...')
        # align reads
        align_reads_experimental(data_directory + f'/GAG_UB_{concentration}.1.sam', data_directory + f'/GAG_UB_{concentration}.2.sam', 
                                 output_directory + '/round1' + f'/aligned_reads_{concentration}_unbound.txt', sequence_length)
        align_reads_experimental(data_directory + f'/GAG_BO_{concentration}.1.sam', data_directory + f'/GAG_BO_{concentration}.2.sam',
                                 output_directory + '/round1' + f'/aligned_reads_{concentration}_bound.txt', sequence_length)
        print('\tFiltering reads...')
        # remove non-significant positions
        remove_non_significant_positions(output_directory + '/round1' + f'/aligned_reads_{concentration}_unbound.txt', 
                                          output_directory + '/round1' + f'/aligned_reads_{concentration}_unbound_filtered.txt', 
                                          significant_positions, position_threshold)
        remove_non_significant_positions(output_directory + '/round1' + f'/aligned_reads_{concentration}_bound.txt',
                                          output_directory + '/round1' + f'/aligned_reads_{concentration}_bound_filtered.txt', 
                                          significant_positions, position_threshold)
        print('\tEncoding reads...')
        # encode mutations
        encode_mutations(output_directory + '/round1' + f'/aligned_reads_{concentration}_unbound_filtered.txt', 
                         output_directory + '/round1' + f'/aligned_reads_{concentration}_unbound_encoded.txt', 
                         path_to_reference_sequence, significant_positions)
        encode_mutations(output_directory + '/round1' + f'/aligned_reads_{concentration}_bound_filtered.txt',
                         output_directory + '/round1' + f'/aligned_reads_{concentration}_bound_encoded.txt', 
                         path_to_reference_sequence, significant_positions)
        print('\tCounting sequences...')
        # count sequences
        count_sequences(output_directory + '/round1' + f'/aligned_reads_{concentration}_bound_encoded.txt', 
                        output_directory + '/round1' + f'/aligned_reads_{concentration}_unbound_encoded.txt', 
                        output_directory + '/round1' + f'/encoded_pool_{concentration}')
        
    # second round:
    for concentration1 in protein_concentrations:
        for concentration2 in protein_concentrations:
            print(f'Processing concentrations: {concentration1} and {concentration2}')
            print('\tAligning reads...')
            # align reads
            align_reads_experimental(data_directory + f'/GAG_UB_{concentration1}_{concentration2}.1.sam', 
                                        data_directory + f'/GAG_UB_{concentration1}_{concentration2}.2.sam', 
                                        output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_unbound.txt', sequence_length)
            align_reads_experimental(data_directory + f'/GAG_BO_{concentration1}_{concentration2}.1.sam',
                                        data_directory + f'/GAG_BO_{concentration1}_{concentration2}.2.sam', 
                                        output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_bound.txt', sequence_length)
            print('\tFiltering reads...')
            # remove non-significant positions
            remove_non_significant_positions(output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_unbound.txt', 
                                                output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_unbound_filtered.txt', 
                                                significant_positions, position_threshold)
            remove_non_significant_positions(output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_bound.txt',
                                                output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_bound_filtered.txt', 
                                                significant_positions, position_threshold)
            print('\tEncoding reads...')
            # encode mutations
            encode_mutations(output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_unbound_filtered.txt', 
                                output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_unbound_encoded.txt', 
                                path_to_reference_sequence, significant_positions)
            encode_mutations(output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_bound_filtered.txt',
                                output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_bound_encoded.txt', 
                                path_to_reference_sequence, significant_positions)
            print('\tCounting sequences...')
            # count sequences
            count_sequences(output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_bound_encoded.txt', 
                            output_directory + '/round2' + f'/aligned_reads_{concentration1}_{concentration2}_unbound_encoded.txt', 
                            output_directory + '/round2' + f'/encoded_pool_{concentration1}_{concentration2}')
    return None

# automated_parsing('/datadisk/MIME/exp/expData', [8, 40, 200, 1000], 2, '/datadisk/MIME/exp/expData/5NL43.fasta', '/datadisk/MIME/exp/expData/sig_pos.txt', '/datadisk/MIME/exp/expData/parsed_data')

align_reads_experimental('/datadisk/MIME/exp/expData/GAG_Wt.1.sam', '/datadisk/MIME/exp/expData/GAG_Wt.2.sam', '/datadisk/MIME/exp/expData/aligned_reads_Wt.txt', 535)
remove_non_significant_positions('/datadisk/MIME/exp/expData/aligned_reads_Wt.txt', '/datadisk/MIME/exp/expData/aligned_reads_Wt_filtered.txt', significant_positions, 2)
encode_mutations('/datadisk/MIME/exp/expData/aligned_reads_Wt_filtered.txt', '/datadisk/MIME/exp/expData/aligned_reads_Wt_encoded.txt', '/datadisk/MIME/exp/expData/5NL43.fasta', significant_positions)