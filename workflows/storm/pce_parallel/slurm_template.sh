#!/bin/bash
#SBATCH --job-name=bout-storm-uq
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --time=0:20:0
#SBATCH --output=bout_storm_uq_%j.log

#SBATCH --account=c01-plasma
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --export=none

#module -s restore PrgEnv-gnu
#module load epcc-job-env
module restore /etc/cray-pe.d/PrgEnv-gnu
module load cray-hdf5
module load cray-netcdf
module load cray-fftw
module load cray-python

np=(128)

#source /work/c01/c01/jtpsla/uq/VECMA-hackathon/venv/bin/activate
#export PYTHONPATH=$PYTHONPATH:/work/c01/c01/jtpsla/uq/VECMA-hackathon/

cd TARGET_DIR
srun --distribution=block:block --hint=nomultithread -n $SLURM_NTASKS /work/c01/c01/jtpsla/uq/VECMA-hackathon/models/storm2d/storm2d -d . -time_report:show
