MODEL_NAME=$1
OUTPUT_DIR=$2

if [ -z ${MODEL_NAME} ]
    then
        echo "No model name specified. Defaulting to bert-base-cased."
        MODEL_NAME="bert-base-cased"
fi

if [ -z ${OUTPUT_DIR} ]
    then
        echo "No output directory specified. Defaulting to /hdd1/seokwon/bluebert_pytorch_outputs/."
        OUTPUT_DIR="/hdd1/seokwon/bluebert_pytorch_outputs"
fi

# Named Entity Recognition ########################################################################
TASK_NAME='NER'
ENTITY_NAME='BC5CDR-chem'

if [ ${TASK_NAME} = 'NER' ]
    then
        TASK_DIR="named-entity-recognition"
        DATA_DIR="../datasets/NER"
fi

ROOT_DIR=$(pwd)

cd ${TASK_DIR}

NER_OUTPUT=$(python ./run_ner.py \
    --log_dir ./results \
    --model_type ${MODEL_NAME} \
    --data_dir ${DATA_DIR}/${ENTITY_NAME} \
    --labels ${DATA_DIR}/${ENTITY_NAME}/labels.txt \
    --model_name_or_path ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR}/${ENTITY} \
    --max_seq_length 128 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --save_steps 1000 \
    --seed 1 \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir)

NER_OUTPUT > "/home/seokwon/testing_script.txt"
###################################################################################################

echo
echo "CHECK OUTPUT"
echo ${NER_OUTPUT}

# export DATA_DIR='../datasets/NER'
# export ENTITY='BC5CDR-chem'

# export MODEL_NAME=bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12

# python ./run_ner.py \
#     --log_dir ./results \
#     --model_type ${MODEL_NAME} \
#     --data_dir ${DATA_DIR}/${ENTITY} \
#     --labels ${DATA_DIR}/${ENTITY}/labels.txt \
#     --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
#     --output_dir ${OUTPUT_DIR}/${ENTITY} \
#     --max_seq_length 128 \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 16 \
#     --save_steps 1000 \
#     --seed 1 \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --overwrite_output_dir
