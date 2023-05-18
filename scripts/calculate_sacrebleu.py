import fire
from sacrebleu import corpus_bleu
from sacremoses import MosesDetokenizer

def main(gen_file, tgt_lang):
    detokenizer = MosesDetokenizer(lang=tgt_lang).detokenize
    tokenize = "intl" if tgt_lang == "de" else "13a"
    refs, hyps = [], []
    with open(gen_file, "r") as f:
        for line in f:
            if line[:2] == "T-":
                refs.append(detokenizer(line.split('\t')[-1].split()))
            if line[:2] == "H-":
                hyps.append(detokenizer(line.split('\t')[-1].split()))
    print(corpus_bleu(hyps, [refs], tokenize=tokenize))
    

if __name__ == '__main__':
    fire.Fire(main)