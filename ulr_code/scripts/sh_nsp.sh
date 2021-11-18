export CUDA_VISIBLE_DEVICES=2,3
export TRAIN_FILE=../datasets/NSP/wiki.en.train
#export TRAIN_FILE=../datasets/SOP
export OUTDIR=../mse_output/nsp
python src/run_nsp.py \
    --output_dir=$OUTDIR \
    --num_train_epochs=1\
    --block_size=128 \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased  \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --mlm \
    --per_device_train_batch_size=16 \
    --logging_steps=10 \
    --save_steps=10000 \
    --logging_dir=$OUTDIR/log_out \
    #--fp16
    #--do_eval \
    #--eval_data_file=$TEST_FILE \
    #--evaluate_during_training \
