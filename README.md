# STAT-520A-Project

## Proposal

A Markov chain is a model that tells us something about the probabilities of sequences of random variables,
states, each of which can take on values from some set (Jurafsky & Martin, 2020). A Hidden Markov Model, or HMM, is a generalization of a Markov chain and allows for analysis on both observed and hidden states (Fonzo et al, 2007). HMMs have a wide variety of applications in the Bioinformatics field. Some prominent applications include: alignment, profiling of sequences, protein structure prediction, and pattern recognition. We propose to create an HMM model to perform single-cell copy-number inference. Utilizing the direct library preparation method (Zahn et al, 2017), we obtain a matrix of read counts which we will perform copy-number alteration (CNA) inference on.

### Works Cited

-Jurafsky, and James Martin. Speech and Language Processing. , 30 Dec. 2020.
-Eddy, Sean R. Multiple Alignment Using Hidden Markov Models. , 1995.
-Zahn, Hans, et al. “Scalable Whole-Genome Single-Cell Library Preparation without           Preamplification.” Nature Methods, vol. 14, no. 2, 1 Feb. 2017, pp. 167–173,              pubmed.ncbi.nlm.nih.gov/28068316/, 10.1038/nmeth.4140. Accessed 11 Feb. 2021.