import subprocess
import yaml


def load(path):
    with open(path) as f:
        return yaml.load(f)


def run_job(job, node, paths):
    # Contruct ENV
    env_script = \
        f"""export PROJ_PATH="{paths["proj_path"]}"
export EXP_NAME="{paths["exp_name"]}"
export SRC_PATH="{paths["src_path"]}"
export XBD_PATH="{paths["xbd_path"]}"
export EXP_PATH="{paths["exp_path"]}"
export DATA_PATH="{paths["data_path"]}"
export OUT_PATH="{paths["out_path"]}"
export FILE_LIST=({str(job["file_list"]).strip("[]").replace(",","").replace("'",'"')})
export CONF_NUM={job["conf_num"]}
"""
    # Construct SLURM script based on job_config
    slurm_script = \
        f"""#!/bin/bash

#SBATCH --job-name="{job['job_name']}"
#SBATCH --output="{paths['proj_path']}/{paths['exp_name']}/out/jobs/{job['job_name']}/out.log"
#SBATCH --error="{paths['proj_path']}/{paths['exp_name']}/out/jobs/{job['job_name']}/out.err"
#SBATCH --nodelist={node['node_name']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={node['cpu_num']}
#SBATCH --partition={node['partition']}
#SBATCH --time={node['hours']}:00:00

/bin/bash -c "{paths['proj_path']}/submit/toko/{job['job_name']}_run.sh"
"""
    run_script = \
        f"""#!/bin/bash

source /home/mcogo/scratch/submit/toko/{job['job_name']}_temp_env.sh
""" + """
output_file="$OUT_PATH/time.txt"

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate develop

# Start timer
start=$(date +%s)

# Run the training script
for file in ${FILE_LIST[@]}; do
    python "${file}"
done

# End timer
end=$(date +%s)
conda deactivate

# Calculate and print execution time
execution_time=$((end - start))
echo "($start,$end) Execution time was ${execution_time} seconds." > "$output_file"
"""
    # write temporal env.sh
    env_file = f"/home/mcogo/scratch/submit/toko/{job['job_name']}_temp_env.sh"
    with open(env_file, 'w') as file:
        file.write(env_script)
    # Write SLURM script to temporary file
    slurm_script_file = f"/home/mcogo/scratch/submit/toko/{job['job_name']}_temp_slurm.sh"
    with open(slurm_script_file, 'w') as file:
        file.write(slurm_script)

    run_script_file = f"/home/mcogo/scratch/submit/toko/{job['job_name']}_run.sh"
    with open(run_script_file, 'w') as file:
        file.write(run_script)

    subprocess.run(['chmod', '777', env_file])
    subprocess.run(['chmod', '777', slurm_script_file])
    subprocess.run(['chmod', '777', run_script_file])
    # Submit job using sbatch
    subprocess.run(['sbatch', slurm_script_file])
    # Clean up temporary file


if __name__ == "__main__":
    config = load('/home/mcogo/scratch/submit/toko/not_aug_job.yaml')
    for job in config['jobs']:
        run_job(job, config['node'], config['paths'])
