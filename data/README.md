## Data frame column interpretation

chr: chromosome number

start: start of bin

end: end of bin

reads: read count

gc: G and C base proportion for bin

map: mappability of bin (higher means more mappable)

valid: Bins with valid GC and average mappability and non-zero read

ideal: Valid bins of high mappability and reads that are not outliers

cor.gc: Readcounts after the first GC correction step

cor.map: cor.gc readcounts after a furthur mappability correction

copy: cor.map (essentially a copy number estimate) transformed into log2 space