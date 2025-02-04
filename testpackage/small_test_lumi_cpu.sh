#!/bin/bash
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH --job-name=ctestpackage
#SBATCH -A project_462000358
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH -c 16                 # CPU cores per task
#SBATCH -n 16                  # number of tasks
#SBATCH --mem=0
#SBATCH --hint=multithread


#If 1, the reference vlsv files are generated
# if 0 then we check the v1
create_verification_files=1

# folder for all reference data 
reference_dir="/scratch/project_462000358/testpackage/"
cd $SLURM_SUBMIT_DIR
#cd $reference_dir # don't run on /proj

bin="/scratch/project_462000358/testpackage/vlasiator_dev_tp"
diffbin="/scratch/project_462000358/testpackage/vlsvdiff_DP"

#compare agains which revision
reference_revision="current"

# threads per job (equal to -c )
t=16
module load LUMI/24.03
module load Boost/1.83.0-cpeGNU-24.03
module load partition/C

#--------------------------------------------------------------------
#---------------------DO NOT TOUCH-----------------------------------
nodes=$SLURM_NNODES
#Carrington has 2 x 16 cores
cores_per_node=128
# Hyperthreading
ht=2
#Change PBS parameters above + the ones here
total_units=$(echo $nodes $cores_per_node $ht | gawk '{print $1*$2*$3}')
units_per_node=$(echo $cores_per_node $ht | gawk '{print $1*$2}')
tasks=$(echo $total_units $t  | gawk '{print $1/$2}')
tasks_per_node=$(echo $units_per_node $t  | gawk '{print $1/$2}')
export OMP_NUM_THREADS=$t

#command for running stuff
#run_command="mpirun -n $tasks -N $nodes "
run_command="srun "
small_run_command="srun -n 1 "
run_command_tools="srun -n 1"

umask 007
# Launch the OpenMP job to the allocated compute node
echo "Running $exec on $tasks mpi tasks, with $t threads per task on $nodes nodes ($ht threads per physical core)"

# Define test
source small_test_definitions.sh
wait
# Run tests
source run_tests.sh
wait 20

