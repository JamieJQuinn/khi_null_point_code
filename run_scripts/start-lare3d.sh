#!/bin/bash

#======================================================
#
# Job script for running a parallel job on a single node
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=standard
#
# Specify project account
#SBATCH --account=mactaggart-aMHD
#
# No. of tasks required (max. of 40)
#SBATCH --ntasks=40
#
# Ensure the node is not shared with another job
#SBATCH --exclusive
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=336:00:00
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

module purge

# choose which version to load (foss 2018a contains the gcc 6.4.0 toolchain & openmpi 2.12)
module load foss/2018a

#=========================================================
# Prologue script to record job details
# Do not change the line below
#=========================================================
/opt/software/scripts/job_prologue.sh 
#----------------------------------------------------------

# Modify the line below to run your program
export OMPI_MCA_btl=openib,self

BINARY=bin/lare3d
module load $MPI_LIB_MODULE
mpirun -np $SLURM_NPROCS $BINARY

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
/opt/software/scripts/job_epilogue.sh 
#----------------------------------------------------------
