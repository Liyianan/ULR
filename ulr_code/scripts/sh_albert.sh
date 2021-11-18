export CUDA_VISIBLE_DEVICES=0,1
export TRAIN_FILE=../datasets/en_ngram/wiki.en.train
export OUTDIR=../output/albert_base
python src/run_mse_albert.py \
    --output_dir=$OUTDIR \
    --num_train_epochs=1\
    --block_size=128 \
    --model_type=albert \
    --model_name_or_path=albert-base-v2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --mlm \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 40 \
    --logging_steps=10 \
    --save_steps=10000 \
    --logging_dir=$OUTDIR/log_out \
