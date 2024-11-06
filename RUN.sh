conda activate develop
source ./src/env.sh
cd ./src
python run_preprocessing.py
python run_parameter_search.py
python run_final_training.py
python run_postprocessing.py
cd ..