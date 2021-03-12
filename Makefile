run_all:
	python run_all.py \
		--root_dir /home/mujeen/works/biobert-pytorch-seanyi \
		--model_name_or_path ${model} \
		--save_file 20210310.txt

#################### Template ##################
run_ner:
	python ./named-entity-recognition/run_ner.py \
		--data_dir /home/mujeen/works/biobert-pytorch-seanyi/datasets/NER/${entity}/ \
		--labels /home/mujeen/works/biobert-pytorch-seanyi/datasets/NER/${entity}/labels.txt \
		--model_name_or_path ${model} \
		--model_type ${model} \
		--max_seq_length 192 \
		--num_train_epochs 3 \
		--per_device_train_batch_size 32 \
		--seed 0 \
		--do_train \
		--do_eval \
		--do_predict \
		--overwrite_output_dir \
		--overwrite_cache \
		--log_dir /hdd1/mujeen/tmp/${name}_${entity} \
		--output_dir /hdd1/mujeen/tmp/${name}_${entity}

run_re:
	python ./named-entity-recognition/run_ner.py \
		--data_dir /home/mujeen/works/biobert-pytorch-seanyi/datasets/NER/${entity}/ \
		--labels /home/mujeen/works/biobert-pytorch-seanyi/datasets/NER/${entity}/labels.txt \
		--model_name_or_path ${model} \
		--model_type ${model} \
		--max_seq_length 192 \
		--num_train_epochs 3 \
		--per_device_train_batch_size 32 \
		--seed 0 \
		--do_train \
		--do_eval \
		--do_predict \
		--overwrite_output_dir \
		--overwrite_cache \
		--log_dir /hdd1/mujeen/tmp/${name}_${entity} \
		--output_dir /hdd1/mujeen/tmp/${name}_${entity}

run_bc5cdr_chemical:
	CUDA_VISIBLE_DEVICES=0 make run_ner model=${model} name=${name} entity=BC5CDR-chem

run_bc5cdr_disease:
	CUDA_VISIBLE_DEVICES=1 make run_ner model=${model} name=${name} entity=BC5CDR-disease

run_gad:
	CUDA_VISIBLE_DEVICES=2 make run_re model=${model} name=${name} entity=GAD

run_euadr:
	CUDA_VISIBLE_DEVICES=3 make run_re model=${model} name=${name} entity=BC5CDR-disease

######################## bc5cdr chemical ##################
run_bc5cdr_chem_bert_cased:
	make run_bc5cdr_chemical \
		model=bert-base-cased \
		name="bert_cased"

run_bc5cdr_chem_bert_uncased:
	make run_bc5cdr_chemical \
		model=bert-base-uncased \
		name="bert_uncased"

run_bc5cdr_chem_bluebert:
	make run_bc5cdr_chemical \
		model="bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12" \
		name="bluebert"
		
run_bc5cdr_chem_biobert_r:
	make run_bc5cdr_chemical \
		model="/hdd1/mujeen/PLMs/bio_bert_retrain_1M" \
		name="biobert_r"

run_bc5cdr_chem_biobert:
	make run_bc5cdr_chemical \
		model=dmis-lab/biobert-base-cased-v1.1 \
		name="biobert"

######################## bc5cdr disease ##################
run_bc5cdr_disease_bert_cased:
	make run_bc5cdr_disease \
		model=bert-base-cased \
		name="bert_cased"

run_bc5cdr_disease_bert_uncased:
	make run_bc5cdr_disease \
		model=bert-base-uncased \
		name="bert_uncased"

run_bc5cdr_disease_bluebert:
	make run_bc5cdr_disease \
		model="bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12" \
		name="bluebert"

run_bc5cdr_disease_biobert_r:
	make run_bc5cdr_disease \
		model="/hdd1/mujeen/PLMs/bio_bert_retrain_1M" \
		name="biobert_r"

run_bc5cdr_disease_biobert:
	make run_bc5cdr_disease \
		model=dmis-lab/biobert-base-cased-v1.1 \
		name="biobert"