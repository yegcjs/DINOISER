while getopts "d:b:o:e:" opt; do
    case "${opt}" in
        d)
            DATASET=${OPTARG}
            ;;
        b)
            DATABIN_PATH=${OPTARG}
            ;;
        o)
            OUTPUT_PATH=${OPTARG}
            ;;
        e)
            EXTRA_ARGS=${OPTARG}
            ;;
        *)
            echo "Invalid arguments" 1>&2; exit 1;
            ;;
    esac
done

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk 'BEGIN{FS=","};{print NF}')

case $DATASET in
    iwslt14)
        BSZ=$((32 * 1024 / $NUM_GPUS ))
        MODEL_ARGS="--arch iwslt_base_postnorm --dropout 0.2 --latent-dim 16"
        DIFF_ARGS="$MODEL_ARGS --max-epoch 2000 --validate-interval 50 --time-sampler clipped"
        ;;
    wmt14)
        BSZ=$((128 * 1024 / $NUM_GPUS ))
        MODEL_ARGS="--arch wmt_base_postnorm --dropout 0.1"
        DIFF_ARGS="$MODEL_ARGS --max-epoch 1000 --validate-interval 10 --time-sampler clipped_s"
        ;;
    wmt16)
        BSZ=$((128 * 1024 / $NUM_GPUS ))
        MODEL_ARGS="--arch wmt_base_postnorm --dropout 0.05"
        DIFF_ARGS="$MODEL_ARGS --max-epoch 600 --validate-interval 50 --time-sampler clipped"
        ;;
esac

mkdir -p $OUTPUT_PATH

# train diffusion
fairseq-train $DATABIN_PATH \
    --user-dir ../src \
    --ddp-backend=legacy_ddp \
    --task diffusion_clm \
    --max-source-positions 256 --max-target-positions 256 --skip-invalid-size-inputs-valid-test \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --update-freq 1 --patience 10 --max-tokens $BSZ \
    --save-dir $OUTPUT_PATH/checkpoints \
    --tensorboard-logdir $OUTPUT_PATH/tensorboard \
    --log-file $OUTPUT_PATH/log.txt \
    --keep-best-checkpoints 5 --no-epoch-checkpoints \
    --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --oracle-length --beam 1 --mbr 1 \
    --solver cedi --denoise-steps 20 --denoise-end 0.99 \
    --criterion diffusion_clm_loss \
    $DIFF_ARGS --scheduler interchangable  --self-conditioning --model-output-type x0 \
    $EXTRA_ARGS

# train length
fairseq-train $DATABIN_PATH \
    --user-dir ../src \
    --ddp-backend=legacy_ddp \
    --task diffusion_clm \
    --max-source-positions 256 --max-target-positions 256 --skip-invalid-size-inputs-valid-test \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --update-freq 1 --max-tokens $BSZ \
    --validate-interval-updates 100 \
    --patience 30 --finetune-from-model $OUTPUT_PATH/checkpoints/checkpoint_best.pt \
    --criterion length_classification_loss \
    --save-dir $OUTPUT_PATH/checkpoints_length \
    --tensorboard-logdir $OUTPUT_PATH/tensorboard_length \
    --log-file $OUTPUT_PATH/log_length.txt \
    --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-best-checkpoints 5 --beam 1 --mbr 1 \
    --solver cedi --denoise-steps 20 --denoise-end 0.99 \
    $MODEL_ARGS --scheduler interchangable --self-conditioning \
    --model-output-type x0 --length-predict-type difference

