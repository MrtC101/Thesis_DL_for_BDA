from collections import defaultdict
import subprocess
import yaml


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_job(job, node, paths, out_path):
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
#SBATCH --output={paths['proj_path']}/{paths['exp_name']}/out/jobs/{job['job_name']}.log
#SBATCH --error={paths['proj_path']}/{paths['exp_name']}/out/jobs/{job['job_name']}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task={node['cpu_num']}
#SBATCH --partition={node['partition']}
#SBATCH --time={node['hours']}-00:00:00
##SBATCH --nodelist={node['node_name']}

/bin/bash -c "{paths['proj_path']}/submit/mendieta/{job['job_name']}_run.sh"
"""
    run_script = \
        f"""#!/bin/bash

source {out_path}/{job['job_name']}_temp_env.sh
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
    env_file = f"{out_path}/{job['job_name']}_temp_env.sh"
    with open(env_file, 'w') as file:
        file.write(env_script)
    # Write SLURM script to temporary file
    slurm_script_file = f"{out_path}/{job['job_name']}_temp_slurm.sh"
    with open(slurm_script_file, 'w') as file:
        file.write(slurm_script)

    run_script_file = f"{out_path}/{job['job_name']}_run.sh"
    with open(run_script_file, 'w') as file:
        file.write(run_script)

    subprocess.run(['chmod', '777', env_file])
    subprocess.run(['chmod', '777', slurm_script_file])
    subprocess.run(['chmod', '777', run_script_file])
    # Submit job using sbatch
    return slurm_script_file
    

if __name__ == "__main__":
    out_path = '/home/mcogo/scratch/submit/test'
    conf = 'not_aug_job.yaml'
    config = load(f"{out_path}/{conf}")
    job_ids = defaultdict(str)
    dependency = None 
    for job in config['jobs']:
        slurm_script_file = run_job(job, config['node'], config['paths'], out_path)

        if str(job["job_name"]).startswith("cv"):
            last_work = job_ids["pre"]
            dependency = f"--dependency=afterok:{last_work}"  
        elif str(job["job_name"]).startswith("defini"):
            last_work = ""
            for key, id in job_ids.items():
                if key.startswith("cv"):
                    last_work+=f"{id}:"
            last_work = last_work.strip(":")
            dependency = f"--dependency=afterok:{last_work}"  
        elif str(job["job_name"]).startswith("post"):
            last_work = f"{job_ids['defini']}" 
            dependency = f"--dependency=afterok:{last_work}"  
            
        if dependency:
            print(dependency)
            command = f'sbatch {dependency} {slurm_script_file}'
            dependency = None
        else:
            command = f'sbatch {slurm_script_file}'
        # Usa subprocess.run para ejecutar el comando y capturar el job ID
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Captura el ID del trabajo actual
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            job_ids[job["job_name"]] = f"{job_id}" 
        else:
            print(f"Error submitting job: {result.stderr}")
            last_work = None
