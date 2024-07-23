# SUBMIT VARS
export JOB_NAME="param_search"
export NODE="toko03"
export CPU_NUM=64
export PARTITION="XL"
export HOURS="72"
export CONF_NUM=2

# RUN VARS
export PROJ_PATH="/home/mrtc101/Desktop/tesina/repo/to_toko"
export EXP_NAME="cpu_exp"
export SRC_PATH="$PROJ_PATH/src"
export DATA_PATH="$PROJ_PATH/$EXP_NAME/data"
export OUT_PATH="$PROJ_PATH/$EXP_NAME/out"
export FILE_LIST=(
    "$SRC_PATH/run_parameter_search.py"
)