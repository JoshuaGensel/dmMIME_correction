# double mutagenesis Mutational Interference Mapping Experiment (dmMIME) correction

Performing MIME for a second round allows to generate a high amount of double mutants to 
subsequently uncover epistatic interactions in a target sequence. This introduces a problem though:
The dissociation constants used in the MIME inference are computed as relative to the wildtype.
This is done by computing the Kd of a mutant in a given position and divide it by the Kd of the of
the wildtype in that position. This works out for a single round of MIME, because the rest of a 
sequence that has a wildtype at a given position will on average have an overall Kd of that sequence
close to wildtype. Therefore the relative Kd for a mutant is relative to the wildtype. In dmMIME 
the unbound fraction is used for a second round of MIME to enrich for double mutants with restored 
function. This subsequently should lead to better structural inference via epistasis calculations.
The problem is that now sequences with a wildtype at a given position do not have approximately an
overall Kd of a full wildtype sequence, but rather there Kds will heavily depend on the protein 
concentrations used in the first round. With a high protein concentration (low selective pressure) 
the unbound fraction is gonna contain only sequences with very high overall Kds. Taking a mutants Kd
relative to this high Kd sequences is going to lead to an underprediction of the relative Kd of that
mutant.

Therefore the problem arises that using different protein concentrations results in different 
inferences from the different "experimental arms" in dmMIME.

This work will investigate in what way the inferences differ, and how to correct the inferred Kd 
values to make accurate function and functional structure predictions using dmMIME.