export SAVE_DIR="/hdd1/seokwon/bluebert_re"
export DATA="ChemProt"
export DATA_DIR="../datasets/RE/${DATA}"
export ENTITY=${DATA}

export MAX_LENGTH=128
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=1000
export SEED=1

python ./run_re.py \
  --task_name ${DATA} \
  --data_dir ${DATA_DIR} \
  --model_name_or_path bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 \
  --max_seq_length ${MAX_LENGTH} \
  --num_train_epochs ${NUM_EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --save_steps ${SAVE_STEPS} \
  --seed ${SEED} \
  --do_train \
  --do_predict \
  --learning_rate 5e-5 \
  --output_dir ${SAVE_DIR}/${ENTITY} \
  --overwrite_output_dir

