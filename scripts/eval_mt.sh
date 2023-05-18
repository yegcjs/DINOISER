BSZ=32
while getopts "b:l:m:c:o:t:" opt; do
    case "${opt}" in
        b)
            DATABIN_PATH=${OPTARG}
            ;;
        l)
            LENGTH_BEAM=${OPTARG}
            ;;
        m)
            MBR=${OPTARG}
            ;;
        c)
            CKPT=${OPTARG}
            ;;
        o)
            OUTPUT_PATH=${OPTARG}
            ;;
        t)
            TGT_LANG=${OPTARG}
            ;;
        *)
            echo "Invalid arguments" 1>&2; exit 1;
            ;;
    esac
done

RESULTS_PATH=$OUTPUT_PATH/LB$LENGTH_BEAM-MBR$MBR

fairseq-generate $DATABIN_PATH \
    --user-dir ../src \
    --gen-subset test \
    --task diffusion_clm \
    --path $CKPT \
    --beam $LENGTH_BEAM --mbr $MBR \
    --max-source-positions 256 --max-target-positions 256 \
    --eval-bleu --remove-bpe \
    --eval-bleu-detok moses --eval-bleu-remove-bpe subword_nmt \
    --skip-invalid-size-inputs-valid-test \
    --solver cedi --denoise-end 0.99 --denoise-steps 20 \
    --batch-size $BSZ --generate-mode \
    --results-path $RESULTS_PATH

python3 calculate_sacrebleu.py --gen_file $RESULTS_PATH/generate-test.txt --tgt_lang $TGT_LANG