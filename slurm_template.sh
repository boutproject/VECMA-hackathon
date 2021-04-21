#!/bin/bash
#SBATCH --job-name=VVUQ
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peter.hill@york.ac.uk
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
#SBATCH --account=PHYS-YPIRSE-2019

module load toolchain/foss/2018b
module load toolchain/gompi/2018b
module load data/netCDF-C++4/4.3.0-foss-2018b

cd TARGET_DIR
mpirun -n $SLURM_NTASKS /mnt/lustre/users/ph781/VECMA-hackathon/build/models/blob2d/blob2d -d . -time_report:show
