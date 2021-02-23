export ENTITY=$1
export MODEL_NAME=$2

# Check whether data type is valid or not. In this setting we're only comparing with BC5CDR-{chem, disease}.
case ${ENTITY} in
    "BC5CDR-chem" | "BC5CDR-disease") ;;
    *) echo "Invalid entity value for NER: ${ENTITY}."
    exit 1
esac

# If model_name_or_path isn't specified, throw error.
if [ -z ${MODEL_NAME} ]
    then
        echo "MODEL_NAME is unset."
        exit 1
fi

export OUTPUT_DIR='/hdd1/seokwon/bluebert_pytorch_outputs/ner'
export HOME_DIR="/home/seokwon/github/biobert-pytorch"
export DATA_DIR="${HOME_DIR}/datasets/NER"
export NER_DIR="${HOME_DIR}/named-entity-recognition"

python ${NER_DIR}/run_ner.py \
    --log_dir ./results \
    --model_type ${MODEL_NAME} \
    --data_dir ${DATA_DIR}/${ENTITY} \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR}/${ENTITY} \
    --max_seq_length 128 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --save_steps 1000 \
    --seed 1 \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir
