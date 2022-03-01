#PBS -N NN_Training    		 # job name
#PBS -l nodes=1:ppn=4:gpus=1     # number of nodes, cores per node, and gpus required
#PBS -l pmem=12gb                # memory per core
#PBS -l walltime=4:00:00         # duration of the job (ex: 15 min)
#PBS -q hive-gpu        	 # queue name (where job is submitted)
#PBS -j oe                       # combine output and error messages into 1 file
#PBS -o ${PBS_JOBID}_log.out      # output file name

homedir="/storage/home/hhive1/amann37"
working_dir=${homedir}"/code/materials_informatics"

nvidia-smi
module load cuda
module load anaconda3
conda activate mat_inf

echo $working_dir
cd $working_dir
pwd

cmd="python run_cnn.py"
echo $cmd; $cmd
