conda activate develop
source ./src/env.sh
python -m run_preprocessing.py
python -m run_paramsearch.py
python -m run_final_model_training.py
python -m run_postprocessing.py