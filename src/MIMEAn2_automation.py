import subprocess
import os

path_to_MIMEAn2 = "~/projects/MIMEAn2/build/MIMEAn2"
working_directory = "/home/user/data_directory/MIME_sim_data/small_sequence_no_error"

#parameters
refSeqFile = "/home/user/data_directory/MIME_sim_data/reference_sequence_small.fasta"
alpha = 0.05
minimumNrCalls = 1
minNumberEstimatableKds = 1
minSignal2NoiseStrength	= 0.01
minMutRate = 5
seqBegin = 1
seqEnd = 100
percOfMaxCov = 0.1
joinErrors = "false"
signThreshold = 0
proteinConcentrations = [.1,1,10]

# go into each directory and run MIMEAn2
for protein_concentration in proteinConcentrations:
    # go into first round directory
    first_dir = working_directory + "/prot" + str(protein_concentration)
    os.chdir(first_dir)
    # check if results directory exists
    if os.path.isdir("results"):
        # if it exists, delete it
        subprocess.call("rm -r results", shell=True)
    # create results directory
    os.mkdir("results")
    # create project.txt in results directory
    with open("results/project.txt", 'w') as f:
        f.write("refSeqFile\t" + str(refSeqFile) + "\n")
        f.write("dataDir\t" + str(first_dir) + "\n")
        f.write("alpha\t" + str(alpha) + "\n")
        f.write("minimumNrCalls\t" + str(minimumNrCalls) + "\n")
        f.write("minNumberEstimatableKds\t" + str(minNumberEstimatableKds) + "\n")
        f.write("minSignal2NoiseStrength\t" + str(minSignal2NoiseStrength) + "\n")
        f.write("minMutRate\t" + str(minMutRate) + "\n")
        f.write("seqBegin\t" + str(seqBegin) + "\n")
        f.write("seqEnd\t" + str(seqEnd) + "\n")
        f.write("percOfMaxCov\t" + str(percOfMaxCov) + "\n")
        f.write("joinErrors\t" + str(joinErrors) + "\n")
        f.write("signThreshold\t" + str(signThreshold) + "\n")
        f.write("selected\t" + str(1) + "\t" + str("<sample_name_1>") + "\t" + str(0) + "\n")
        f.write("nonselected\t" + str(2) + "\t" + str("<sample_name_1>") + "\t" + str(0) + "\n")
        f.write("selected\t" + str(3) + "\t" + str("<sample_name_1>") + "\t" + str(1) + "\n")
        f.write("nonselected\t" + str(4) + "\t" + str("<sample_name_1>") + "\t" + str(1) + "\n")

    

    # run MIMEAn2
    subprocess.call(path_to_MIMEAn2 + " " + first_dir + "/results", shell=True)

    # go into second round directory
    for protein_concentration2 in proteinConcentrations:
        second_dir = working_directory + "/secondFromProt" + str(protein_concentration) + "/prot" + str(protein_concentration2)
        os.chdir(second_dir)
        # check if results directory exists
        if os.path.isdir("results"):
            # if it exists, delete it
            subprocess.call("rm -r results", shell=True)
        # create results directory
        os.mkdir("results")
        # copy wildtrype count files into second round directory
        subprocess.call("cp " + first_dir + "/1d/1.txt " + second_dir + "/1d/1.txt", shell=True)
        subprocess.call("cp " + first_dir + "/2d/1.txt " + second_dir + "/2d/1.txt", shell=True)
        subprocess.call("cp " + first_dir + "/1d/2.txt " + second_dir + "/1d/2.txt", shell=True)
        subprocess.call("cp " + first_dir + "/2d/2.txt " + second_dir + "/2d/2.txt", shell=True)
        
        # create project.txt in results directory
        with open("results/project.txt", 'w') as f:
            f.write("refSeqFile\t" + str(refSeqFile) + "\n")
            f.write("dataDir\t" + str(second_dir) + "\n")
            f.write("alpha\t" + str(alpha) + "\n")
            f.write("minimumNrCalls\t" + str(minimumNrCalls) + "\n")
            f.write("minNumberEstimatableKds\t" + str(minNumberEstimatableKds) + "\n")
            f.write("minSignal2NoiseStrength\t" + str(minSignal2NoiseStrength) + "\n")
            f.write("minMutRate\t" + str(minMutRate) + "\n")
            f.write("seqBegin\t" + str(seqBegin) + "\n")
            f.write("seqEnd\t" + str(seqEnd) + "\n")
            f.write("percOfMaxCov\t" + str(percOfMaxCov) + "\n")
            f.write("joinErrors\t" + str(joinErrors) + "\n")
            f.write("signThreshold\t" + str(signThreshold) + "\n")
            f.write("selected\t" + str(1) + "\t" + str("<sample_name_1>") + "\t" + str(0) + "\n")
            f.write("nonselected\t" + str(2) + "\t" + str("<sample_name_1>") + "\t" + str(0) + "\n")
            f.write("selected\t" + str(7) + "\t" + str("<sample_name_1>") + "\t" + str(1) + "\n")
            f.write("nonselected\t" + str(8) + "\t" + str("<sample_name_1>") + "\t" + str(1) + "\n")

        # run MIMEAn2
        subprocess.call(path_to_MIMEAn2 + " " + second_dir + "/results", shell=True)