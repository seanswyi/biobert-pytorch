#!/bin/bash

ENTITIES=ChemProt
MAX_LENGTH=128

for ENTITY in $ENTITIES
do
    echo "***** " $ENTITY " Preprocessing Start *****"
    DATA_DIR=../datasets/RE/$ENTITY

    mv $DATA_DIR/train.tsv $DATA_DIR/train_original.tsv
    mv $DATA_DIR/dev.tsv $DATA_DIR/dev_original.tsv
    mv $DATA_DIR/test.tsv $DATA_DIR/test_original.tsv

    python scripts/preprocess_ddi.py $DATA_DIR/train_original.tsv train  > $DATA_DIR/train.tsv
    python scripts/preprocess_ddi.py $DATA_DIR/dev_original.tsv dev  > $DATA_DIR/dev.tsv
    python scripts/preprocess_ddi.py $DATA_DIR/test_original.tsv test  > $DATA_DIR/test.tsv

done
echo "***** " $ENTITY " Preprocessing Done *****"

