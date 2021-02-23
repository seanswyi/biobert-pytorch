export DATA_DIR=../datasets/NER
export OUTPUT_DIR=/hdd1/seokwon/bluebert_pytorch_outputs/ner

for entity in 'BC5CDR-chem' 'BC5CDR-disease'
do
    for model_type in 'base-pubmed' 'base-pubmed-mimic' 'base'
    do
        python run_ner.py \
            --log_dir ./results \
            --model_type ${model_type} \
            --data_dir ${DATA_DIR}/${entity} \
            --labels ${DATA_DIR}/${entity}/labels.txt \
            --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
            --output_dir ${OUTPUT_DIR}/${entity} \
            --max_seq_length 128 \
            --num_train_epochs 3 \
            --per_device_train_batch_size 32 \
            --save_steps 1000 \
            --seed 1 \
            --do_eval \
            --do_predict \
            --overwrite_output_dir
    done
done
