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




# align_reads_experimental('./data/test_files/experimental/left.1.sam', './data/test_files/experimental/right.2.sam', './data/test_files/experimental/aligned_reads.txt', 536)


# align_reads_simulator('./data/test_files/7.txt', './data/test_files/aligned_reads_bound.txt', 1/20, 7, 42)
# align_reads_simulator('./data/test_files/8.txt', './data/test_files/aligned_reads_unbound.txt', 1/20, 7, 42)


