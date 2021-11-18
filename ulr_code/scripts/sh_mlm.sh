export CUDA_VISIBLE_DEVICES=2,3
export TRAIN_FILE=../datasets/en_ngram/wiki.en.train
export OUTDIR=../output/mlm_base
python src/run_mlm.py \
    --output_dir=$OUTDIR \
    --num_train_epochs=1\
    --block_size=128 \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --mlm \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 36 \
    --logging_steps=10 \
    --save_steps=10000 \
    --logging_dir=$OUTDIR/log_out \
