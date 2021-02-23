export SAVE_DIR="/hdd1/seokwon/bluebert_outputs"
export MODEL_DIR="/hdd1/seokwon/models"

for model in "bluebert-base-uncased-evidence-1.5k" "bluebert-base-uncased-random-1.5k"
    for data in "BC5CDR-chem" "BC5CDR-disease" "DDI" "ChemProt" "MedNLI"
    do
        if [ ${data} == "BC5CDR-chem" ] || [ ${data} == "BC5CDR-disease" ]
        then

            export TASK="NER";
            export DATA_DIR="./datasets/${TASK}/${data}"
            export OUTPUT_DIR="/hdd1/seokwon/bluebert_ner"

	    python ./named-entity-recognition/run_ner.py \
	        --log_dir ./results \
	        --data_dir ${DATA_DIR} \
	        --labels ${DATA_DIR}/labels.txt \
	        --model_name_or_path ${MODEL_DIR}/${model} \
                --output_dir ${OUTPUT_DIR} \
                --max_seq_length 128 \
                --num_train_epochs 3 \
                --per_device_train_batch_size 16 \
                --save_steps 1000 \
                --seed 1 \
                --do_train \
                --do_eval \
                --do_predict \
                --overwrite_output_dir

	elif [ ${data} == "DDI" ] || [ ${data} == "ChemProt" ]
	then

	    export TASK="RE"
            export DATA_DIR="./datasets/${TASK}/${data}"
            export OUTPUT_DIR="/hdd1/seokwon/bluebert_re"

            python ./relation-extraction/run_re.py \
              --task_name ${data} \
              --data_dir ${DATA_DIR} \
              --model_name_or_path ${MODEL_DIR}/${model} \
              --max_seq_length 128 \
              --num_train_epochs 3 \
              --per_device_train_batch_size 32 \
              --save_steps 1000 \
              --seed 1 \
              --do_train \
              --do_eval \
              --do_predict \
              --overwrite_output_dir \
              --overwrite_cache

	elif [ ${data} == "MedNLI" ]
	then

	    export TASK="NLI"
            export DATA_DIR="./datasets/${TASK}/${data}"
            export OUTPUT_DIR="/hdd1/seokwon/bluebert_nli"

            python ./natural-language-inference/run_nli.py \
                --task_name mednli \
                --data_dir ${DATA_DIR} \
                --model_name_or_path ${MODEL_DIR}/${model} \
                --max_seq_length 128 \
                --num_train_epochs 3 \
                --per_device_train_batch_size 3 \
                --save_steps 32 \
                

	fi

    
