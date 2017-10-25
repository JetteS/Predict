## source /com/extra/Anaconda-Python/2.2.0-2.7/load.sh
from __future__ import print_function
import h5py
import plinkio
from plinkio import plinkfile
import scipy as sp
from itertools import izip


def bed_to_hdf5_file(bed_file,hdf5_out):

	"""
	Note: It may not support all PLINK files for now
	"""

	plinkf = plinkfile.PlinkFile(bed_file)
	samples = plinkf.get_samples()
	print("Extracting sample information...")

	affections = []
	phenotypes = []
	iids = []
	fids = []
	sex = []
	
	## For each sample extract the individual identifier, the family identifier,
	## the affection, the phenotype and the sex.
	for sample in samples:
		iids.append(sample.iid)
		fids.append(sample.fid)
		affections.append(sample.affection)
		#phenotypes.append(sample.phenotype)
		sex.append(sample.sex)

	## Number of individuals
	N = len(iids)

	if sp.any(sp.isnan(phenotypes)):
		print('Phenotypes appear to have some NaNs, or perhaps parsing failed?')
	else:
		print("%d individuals have phenotype and genotype information." % N)

	hf = h5py.File(hdf5_out)

	## Store sample information in HDF5 file
	sample_inf = hf.create_group('sample_informations')
	sample_inf.create_dataset('iids', data=iids)
	sample_inf.create_dataset('fids', data=fids)
	sample_inf.create_dataset('Affections', data=affections)
	sample_inf.create_dataset('Sex', data=sex)
	sample_inf.create_dataset('Phenotypes', data=phenotypes)
	hf.flush()

	print("Iterating over BED file...")

	## Iterate through the plink file.
	locus_list = plinkf.get_loci()
	chromosomes = []
	current_chromosome = 0
	print("The current chromosome is Chr", current_chromosome)

	for locus, row in izip(locus_list, plinkf):
		## Get the current chromosome
		chrom = locus.chromosome
		## and store it in the chromosome vector
		chromosomes.append(locus.chromosome)
		if current_chromosome == 0:
			## Initialize data containers
			sids = []
			positions = []
			nts_list = []
			snps = []
		if chrom != current_chromosome:
			## Print the number of the chromosome
			print("The current chromosome is Chr", chrom)
			# Store current data in the HDF5 file
			chr_group = hf.create_group("chr_%d" % current_chromosome)
			chr_group.create_dataset("sids", data = sids)
			chr_group.create_dataset('positions', data=positions)
		 	chr_group.create_dataset('snps', data=sp.array(snps, dtype='int8'))
		 	chr_group.create_dataset('nts_list', data=nts_list)
		 	hf.flush()

		 	## re-initialize data containers
		 	sids = []
			positions = []
			nts_list = []
			snps = []
			current_chromosome = chrom

		## Get the SNP name
		sids.append(locus.name)
		## Get the first and the second allele 
		nts_list.append([locus.allele1, locus.allele2])
		## Furthermore, we store the position 
		positions.append(locus.position)

		## Parse SNP and fill in the blanks if necessary
		snp = sp.array(row, dtype="int8")
		bin_counts = row.allele_counts()
		if bin_counts[-1] > 0:
			mode_v = sp.argmax(bin_counts[:2])
			snp[snp == 3] = mode_v

		snps.append(snp)

	## Store remaining data in HDF5 file
	chr_group = hf.create_group("chr_%d" % current_chromosome)
	chr_group.create_dataset("sids", data = sids)
	chr_group.create_dataset('positions', data=positions)
	chr_group.create_dataset('snps', data=sp.array(snps, dtype='int8'))
	chr_group.create_dataset('nts_list', data=nts_list)
	hf.create_dataset("Chromosomes", data = chromosomes)
	hf.flush()

	plinkf.close()
	hf.close()
	print("The parsing is completed")

#bed_to_hdf5_file("../risk_prediction/faststorage/celiac_disease_data/Cel_disease_CC","First_try.h5")
