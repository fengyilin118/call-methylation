# Nanopolish GPU-implementation Call-methylation


## Usage

```sh
f5c index -d [fast5_folder] [read.fastq|fasta]
f5c call-methylation -b [reads.sorted.bam] -g [ref.fa] -r [reads.fastq|fasta] > [meth.tsv]
f5c meth-freq -i [meth.tsv] > [freq.tsv]
f5c eventalign -b [reads.sorted.bam] -g [ref.fa] -r [reads.fastq|fasta] > [events.tsv]    #specify --rna for direct RNA data
```
