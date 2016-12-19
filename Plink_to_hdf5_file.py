from __future__ import print_function
import h5py
from plinkio import plinkfile

def plink_to_hdf5(tped_file, tfam_file, hdf5_file, N, M):

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

	with open(tped_file, "r") as pf, open(tfam_file, "r") as ff, h5py.File(hdf5_file, "w") as hf:
		Genotypes = hf.create_dataset("Genotypes", (M, N), dtype="i8")
		Chromosome = hf.create_dataset("Chromosome", (M, 1), dtype="i8")
		Phenotype = hf.create_dataset("Phenotype", (N, 1), dtype="i8")
		i = 0
		for line in pf:
			fields = line.split()
			Chromosome[i] = fields[0]
			data = fields[4:]
			gt = []
			for i in range(0, len(data), 2):
				h0 = int(data[i])
				h1 = int(data[i + 1])
				if h0 != 0 and h1 != 0:
					gt.append(h0 + h1 - 2)
				else:
					# del data[i:i+2]
					# gt.append(mean(data))
					gt.append(0)
			
			Genotypes[i] = gt
			i += 1

		i = 0
		for line in ff:
			fam = line.split()
			Phenotype[i] = fam[0]
			i += 1

		pass 


def bed_plink_to_hdf5(genotype_file, out_hdf5_file):
	"""
	Note: It may not support all PLINK files for now.
	"""
	
	plinkf = plinkfile.PlinkFile(genotype_file)
	samples = plinkf.get_samples()
	
	affections = []
	phens = []
	iids = []
	fids = []
	
	for samp_i, sample in enumerate(samples):
		iids.append(sample.iid)
		fids.append(sample.fid)
		affections.append(sample.affection)
		phens.append(sample.phenotype)
				
	num_individs = len(iids)
	if sp.any(sp.isnan(true_phens)):
		print('Phenotypes appear to have some NaNs, or perhaps parsing failed?')
	else:
		print('%d individuals have phenotype and genotype information.' % num_individs)
	
	# If these indices are not in order then we place them in the right place while parsing SNPs.
	print('Iterating over BED file.')
	oh5f = h5py.File(out_hdf5_file)
	# First construct chromosome groups.
	
	# Then iterate through the plink file.
	locus_list = plinkf.get_loci()
	snp_i = 0
	for locus, row in it.izip(locus_list, plinkf):
		sid = locus.name
		nts = [locus.allele1, locus.allele2]

		# Parse SNP, and fill in the blanks if necessary.
		snp = sp.array(row, dtype='int8')[indiv_filter]
		bin_counts = row.allele_counts()
		if bin_counts[-1] > 0:
			mode_v = sp.argmax(bin_counts[:2])
			snp[snp == 3] = mode_v
		
		# Store data in HDF5 file

	plinkf.close()


plink_to_hdf5("../Transpose_data/QC_PASS.CD.1trans.tped", "../Transpose_data/QC_PASS.CD.1trans.tfam", "First_try.h5")
