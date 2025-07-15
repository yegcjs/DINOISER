# DINOISER: Diffused Conditional Sequence Learning by Manipulating Noises

This repository contains the official implementation of paper [DINOISER: Diffused Conditional Sequence Learning by Manipulating Noises (TACL & ACL2024 Oral)](https://arxiv.org/abs/2302.10025).

---
## Dependencies

The code is implemented with fairseq. To setup the dependencies, run
```bash
pip3 install -r requirements.txt
```

## Data Preparation

The following commands utilize the [script](https://github.com/facebookresearch/fairseq/blob/main/examples/translation/prepare-iwslt14.sh) from fairseq's examples to prepare `IWSLT14 En<->De` data.
```bash
mkdir iwslt14
cd iwslt14

wget -O - https://raw.githubusercontent.com/facebookresearch/fairseq/main/examples/translation/prepare-iwslt14.sh | bash

fairseq-preprocess -s en -t de \
    --trainpref iwslt14.tokenized.de-en/train \
    --validpref iwslt14.tokenized.de-en/valid \
    --testpref iwslt14.tokenized.de-en/test \
    --destdir iwslt14.en-de.real.bin
    --workers 32

fairseq-preprocess -s de -t en \
    --trainpref iwslt14.tokenized.de-en/train \
    --validpref iwslt14.tokenized.de-en/valid \
    --testpref iwslt14.tokenized.de-en/test \
    --destdir iwslt14.de-en.real.bin
    --workers 32

cd ..
```

For `WMT14 En<->De` and `WMT16 En<->Ro`, we obtain the preprocessed data from the links that [Fully-NAT](https://github.com/shawnkx/Fully-NAT#dataset) provides.

```
wget https://dl.fbaipublicfiles.com/nat/fully_nat/datasets/wmt14.en-de.zip
wget https://dl.fbaipublicfiles.com/nat/fully_nat/datasets/wmt16.ro-en.zip
unzip wmt14.en-de -d wmt14
unzip wmt16.ro-en -d wmt16
```

## Training and Evaluation

Use the follwoing scripts to train the models for machin translation.

```bash
cd scripts
CUDA_VISIBLE_DEVICES=0 bash train_mt.sh -d iwslt14 -b ../iwslt14/iwslt14.de-en.real.bin -o ../outputs/iwslt14.de-en
CUDA_VISIBLE_DEVICES=0 bash train_mt.sh -d iwslt14 -b ../iwslt14/iwslt14.en-de.real.bin -o ../outputs/iwslt14.en-de
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_mt.sh -d wmt14 -b ../wmt14/wmt14.de-en.real.bin -o ../outputs/wmt14.de-en
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_mt.sh -d wmt14 -b ../wmt14/wmt14.en-de.real.bin -o ../outputs/wmt14.en-de
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_mt.sh -d wmt16 -b ../wmt16/wmt16.ro-en.real.bin -o ../outputs/wmt16.ro-en
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_mt.sh -d wmt16 -b ../wmt16/wmt16.en-ro.real.bin -o ../outputs/wmt16.en-ro
```
* `-d`: Identifier for dataset. We use it to setup dataset specific arguments (e.g., model architecture and batch size) in the script.
* `-b`: Directory of the binarize data.
* `-o`: Directory for saving checkpoints and logs. 
* You can append `-e "--bf16"` to accelerate training if your devices support bfloat16.


Then, use the following scipts to make inferences and evaluate the models.

```bash
bash eval_mt.sh -b ../iwslt14/iwslt14.de-en.real.bin -l 10 -m 5 -c ../outputs/iwslt14.de-en/checkpoints_length/checkpoint_best.pt -o ../outputs/iwslt14.de-en -t en
bash eval_mt.sh -b ../iwslt14/iwslt14.en-de.real.bin -l 10 -m 5 -c ../outputs/iwslt14.en-de/checkpoints_length/checkpoint_best.pt -o ../outputs/iwslt14.en-de -t de
bash eval_mt.sh -b ../wmt14/wmt14.de-en.real.bin -l 10 -m 5 -c ../outputs/wmt14.de-en/checkpoints_length/checkpoint_best.pt -o ../outputs/wmt14.de-en -t en
bash eval_mt.sh -b ../wmt14/wmt14.en-de.real.bin -l 10 -m 5 -c ../outputs/wmt14.en-de/checkpoints_length/checkpoint_best.pt -o ../outputs/wmt14.en-de -t de
bash eval_mt.sh -b ../wmt16/wmt16.de-en.real.bin -l 10 -m 5 -c ../outputs/wmt16.ro-en/checkpoints_length/checkpoint_best.pt -o ../outputs/wmt16.ro-en -t en
bash eval_mt.sh -b ../wmt16/wmt16.de-en.real.bin -l 10 -m 5 -c ../outputs/wmt15.de-en/checkpoints_length/checkpoint_best.pt -o ../outputs/wmt16.en-ro -t ro
```
Arguments
* `-b`: Directory of the binarize data.
* `-l`: Length beam.
* `-m`: Number of MBR candidates for each length beam.
* `-c`: The checkpoint to be evaluated.
* `-o`: Directory to place the generation result.
* `-t`: The language of the target. This affects the tokenizer for computing sacrebleu.

Please check the scripts for more details.

## Checkpoints

|Dataset|Checkpoint|
|-------|----------|
|IWSLT14 De->En|[Download](https://box.nju.edu.cn/f/0caa95073bc04d198f50/?dl=1)|
|IWSLT14 En->De|[Download](https://box.nju.edu.cn/f/970b5ff1c7b54fb38a15/?dl=1)|
|WMT14 De->En|[Download](https://box.nju.edu.cn/f/1310fcbf7de84721918f/?dl=1)|
|WMT14 En->De|[Download](https://box.nju.edu.cn/f/0a7e51ed22b84b9e8e7a/?dl=1)|
|WMT16 Ro->En|[Download](https://box.nju.edu.cn/f/ec0c5bc8ff78426e8118/?dl=1)|
|WMT16 En->Ro|[Download](https://box.nju.edu.cn/f/c9af0cfd20a4431fb016/?dl=1)|

## Citation
```
@article{ye2024dinoiser,
  title={DINOISER: Diffused Conditional Sequence Learning by Manipulating Noises},
  author={Ye, Jiasheng and Zheng, Zaixiang and Bao, Yu and Qian, Lihua and Wang, Mingxuan},
  journal={Transaction of ACL},
  year={2024}
}
```
