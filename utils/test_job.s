#!/bin/bash -l
#SBATCH --job-name test_pulled_vesicle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --mail-user=llu@flatironinstitute.org
#SBATCH --output=one_pulled_vesicle_%A_%a.out
#SBATCH --partition=ccm
#SBATCH --constraint=skylake

SRCDIR=/mnt/home/llu/projects/ves3d

cd $SRCDIR
source utils/rusty_modules.sh

RUNDIR=/mnt/ceph/users/$USER/ves3d/pulled_vesicle-$SLURM_JOB_ID-${SLURM_ARRAY_TASK_ID/.*}
mkdir -p $RUNDIR
mkdir -p $RUNDIR/result

cd $SLURM_SUBMIT_DIR
cp experiment/one_pulled_vesicle.in $RUNDIR
cp -rf precomputed $RUNDIR
cd $RUNDIR

if [ "$SLURM_JOBTMP" == "" ]; then
    export SLURM_JOBTMP=$RUNDIR/$$
    mkdir -p $SLURM_JOBTMP
fi

###
DIM1=5
DIM2=4

bm[0]=0.1
bm[1]=0.025
bm[2]=0.005
bm[3]=0.001
bm[4]=0.0001

mind[0]=0.08
mind[1]=0.04
mind[2]=0.02
mind[3]=0.01


i=$((SLURM_ARRAY_TASK_ID/DIM1))
j=$((SLURM_ARRAY_TASK_ID-i*DIM1))

bending=${bm[j]}
minsep=${mind[i]}
###


echo 
echo "Job starts: $(date)"
echo "Hostname: $(hostname)"
echo

cat<<EOF | $SRCDIR/bin/pulled_vesicle -f one_pulled_vesicle.in > log

EOF

exe_status=$?;

echo 
echo "Job ends: $(date) exe_status: $(exe_status)"
echo

exit $exe_status
