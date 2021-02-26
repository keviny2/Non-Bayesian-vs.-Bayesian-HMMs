## TODO

- create a pipeline to load data

    - create matrix with a single row representing chromosome, columns representing bins and values are read counts
    

## Notes

- We don't know the transition probabilities and emission probabilities, 
so need to use either the Viterbi learning or Baum-Welch algo
  
- Need to choose a good number of states for CNA (HMMCopy uses 6, see vignette)