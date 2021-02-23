export ENTITY=$1
export MODEL_NAME=$2

# Check whether data type is valid or not. In this setting we're only using MedNLI.
case ${ENTITY} in
    "MedNLI") ;;
    *) echo "Invalid entity value for NLI: ${ENTITY}."
    exit 1
esac

# If model_name_or_path isn't specified, throw error.
if [ -z ${MODEL_NAME} ]
    then
        echo "MODEL_NAME is unset."
        exit 1
fi

export OUTPUT_DIR="/hdd1/seokwon/bluebert_pytorch_outputs/nli"
export HOME_DIR="/home/seokwon/github/biobert-pytorch"
export DATA_DIR="${HOME_DIR}/datasets/NLI/${ENTITY}"
export NLI_DIR="${HOME_DIR}/natural-language-inference"

python ${NLI_DIR}/run_nli.py \
    --task_name ${ENTITY} \
    --data_dir ${DATA_DIR} \
    --model_name_or_path ${MODEL_NAME} \
    --max_seq_length 128 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --save_steps 1000 \
    --seed 1 \
    --do_train \
    --do_predict \
    --learning_rate 5e-5 \
    --output_dir ${OUTPUT_DIR}/${ENTITY} \
    --overwrite_output_dir \
    --overwrite_cache