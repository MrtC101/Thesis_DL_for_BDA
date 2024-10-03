from collections import defaultdict
import subprocess
import yaml
import os


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def environment_script(paths):
    return f"""export PROJ_PATH="{paths["proj_path"]}"
export EXP_NAME="{paths["exp_name"]}"
export SRC_PATH="{paths["src_path"]}"
export XBD_PATH="{paths["xbd_path"]}"
export EXP_PATH="{paths["exp_path"]}"
export DATA_PATH="{paths["data_path"]}"
export OUT_PATH="{paths["out_path"]}"
export FILE_LIST=({str(job["file_list"]).strip("[]").replace(",","").replace("'",'"')})
export CONF_NUM={job["conf_num"]}
"""


def python_run_script(out_path, job):
    return f"""#!/bin/bash

source {out_path}/{job['job_name']}_temp_env.sh
""" + """
output_file="$OUT_PATH/time.txt"

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate develop

# Run the training script
for file in ${FILE_LIST[@]}; do
    python "${file}"
done
"""


def slurm_script(paths, node, job):
    return f"""#!/bin/bash

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

    /bin/bash -c "{out_path}/{job['job_name']}_run.sh"
    """


def create_temp_files(job: dict, node: str, paths: dict, out_path: str) -> str:
    """
    Creates 3 temporary files based on the job specifications: 
    - Environment script (env.sh)
    - Bash script to run a Python file
    - SLURM job script

    Args:
        job (dict): Job specifications.
        node (str): Node information for the SLURM script.
        paths (dict): Paths for environment setup.
        out_path (str): Directory where temporary files will be saved.

    Returns:
        str: Path to the SLURM script file.
    """
    # Permission for scripts
    script_permissions = '777'

    # Helper function to set file permissions
    def set_file_permissions(file_path: str, permissions: str):
        subprocess.run(['chmod', permissions, file_path])

    # Create environment script (env.sh)
    env_file = os.path.join(out_path, f"{job['job_name']}_temp_env.sh")
    with open(env_file, 'w') as file:
        file.write(environment_script(paths))
    set_file_permissions(env_file, script_permissions)

    # Create bash script to run the Python file
    run_script_file = os.path.join(out_path, f"{job['job_name']}_run.sh")
    with open(run_script_file, 'w') as file:
        file.write(python_run_script(out_path, job))
    set_file_permissions(run_script_file, script_permissions)

    # Create SLURM job script
    slurm_script_file = os.path.join(out_path, f"{job['job_name']}_temp_slurm.sh")
    with open(slurm_script_file, 'w') as file:
        file.write(slurm_script(paths, node, job))
    set_file_permissions(slurm_script_file, script_permissions)

    return slurm_script_file


def run_job(job: dict, job_ids: dict[str, list], slurm_script_file: str) -> dict[str, list]:
    """Ejecuta el trabajo SLURM y actualiza job_ids con el ID del nuevo trabajo."""

    prefix_to_dependency = {
        "pre": "",
        "cv": "pre",
        "final": "cv",
        "post": "post"
    }

    job_name = str(job["job_name"])
    # Añade las dependencias
    dependency = ""
    for key, dep in prefix_to_dependency:
        if job_name.startswith(key) and len(job_ids[dep]) > 0:
            dependency = "--dependency=afterok"
            for id in job_ids[dep]:
                dependency += f":{id}"

    print(f"{job_name}: {dependency}")
    command = f'sbatch {dependency} {slurm_script_file}'

    # Usa subprocess.run para ejecutar el comando y capturar el job ID
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Captura el ID del trabajo actual
    if result.returncode == 0:
        queue_id = result.stdout.strip().split()[-1]
        for key in prefix_to_dependency.keys():
            if job_name.startswith(key):
                job_ids[key].append(queue_id)
                break
    else:
        print(f"Error submitting job: {result.stderr}")

    return job_ids


if __name__ == "__main__":
    # Arguments
    out_path = '/home/mcogo/scratch/submit/mendieta/exp5'
    config = load(f"{out_path}/exp5_job.yaml")
    run = False

    job_ids = defaultdict(list)
    for job in config['jobs']:
        slurm_script_file = create_temp_files(job, config['node'], config['paths'], out_path)
        if run:
            job_ids = run_job(job_ids)
