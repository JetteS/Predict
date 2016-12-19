from __future__ import print_function
import h5py

def plink_to_hdf5(tped_file, tfam_file, hdf5_file,N,M):

# Write snps from tped and tfam file to hdf5 file.
# NOTE: The snps has to be encoded in the following way:
# 0.. missing
# 1.. unaffected
# 2.. affected
# Such an encoding can be obtained by using the command 
# --recode12 in Plink.
# This function assumes that all individuals are diploid.
# The two haploid columns in the tped file become a single 
# genotype in the hdf5 file.

	with open(tped_file,"r") as pf, open(tfam_file,"r") as ff, h5py.File(hdf5_file,"w") as hf:
		Genotypes = hf.create_dataset("Genotypes", (M,N), dtype="i8")
		Chromosome = hf.create_dataset("Chromosome", (M,1), dtype="i8")
		Phenotype = hf.create_dataset("Phenotype", (N,1), dtype="i8")
		i=0
		for line in pf:
			fields = line.split()
			Chromosome[i]=fields[0]
			data = fields[4:]
			gt = []
			for i in range(0,len(data),2):
				h0 = int(data[i])
				h1 = int(data[i+1])
				if h0!=0 and h1!=0:
					gt.append(h0+h1-2)
				else:
					#del data[i:i+2]
					#gt.append(mean(data))
					gt.append(0)
			
			Genotypes[i] = gt
			i+=1

		i=0
		for line in ff:
			fam = line.split()
			Phenotype[i] = fam[0]
			i+=1

		pass 


plink_to_hdf5("../Transpose_data/QC_PASS.CD.1trans.tped", "../Transpose_data/QC_PASS.CD.1trans.tfam", "First_try.h5")
