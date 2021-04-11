# RuSimpleSentEval
This repo contains some code and a model for the first place solution of [RuSimpleSentEval](https://github.com/dialogue-evaluation/RuSimpleSentEval).

Play with it: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1izjYmxWYO6mTWt5XEQ06dV32yMjfKsML?usp=sharing#forceEdit=true&sandboxMode=true)

It is actually highly based on the ideas from [Multilingual Unsupervised Sentence Simplification](https://arxiv.org/abs/2005.00352).

Basically, it's mBART finetuned on the paraphrases from [ParaPhraserPlus](http://paraphraser.ru/download/) and [automatically translated WikiLarge](https://github.com/dialogue-evaluation/RuSimpleSentEval#данные) conditioned on some control tokens.

The control tokens give us ability to train the model on everything that is semantically related, and then to choose those control token values which work the best for simplification (according to some metric).

The following control tokens were implemented:
- Levenshtein similarity - how similar (obviously by levenshtein metric) the result text should be;
- Chars fraction - how long the result text should be (that is, the you can specify the ratio between the result and original text lengths);
- Word rank - how simple the result text is expected to be (it's a ratio again between the ranks of the words in the texts in [fasttext embeddings](http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html#id4)). Didn't work well in my opinion;
- Lexeme similarity - how similar by lexeme matching the result text should be. Wasn't used in the final model, though.

## Downloads
- [Fairseq checkpoint](https://drive.google.com/file/d/1eD55lwl0X1Bf2-hgXGi3-K7vmnngpDHr/view?usp=sharing)
- [Huggingface checkpoint + control token mapping](https://drive.google.com/file/d/1WfEq9Jfqi9sQQZXzDA3_MSHVmccbGgX-/view?usp=sharing)
- [Preprocessed train data + some files used for preprocessing](https://drive.google.com/file/d/17y_zS1N7agrDiSwZtnEPmPu8NzgJ2KPb/view?usp=sharing)

## Train
The training process consist of the following steps.

## Requirements Installation
The instructions are based on the [baseline solution](https://github.com/dialogue-evaluation/RuSimpleSentEval#базовое-решение).

1. Install SentencePiece:
```bash
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig -v
```

2. Install fairseq (current pip version doesn't have all required features, but it should change at some point):
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

Also I added the following line:
```diff
diff --git a/fairseq/tasks/translation_from_pretrained_bart.py b/fairseq/tasks/translation_from_pretrained_bart.py
index 8710b7f..a7ff1db 100644
--- a/fairseq/tasks/translation_from_pretrained_bart.py
+++ b/fairseq/tasks/translation_from_pretrained_bart.py
@@ -51,6 +51,7 @@ class TranslationFromPretrainedBARTTask(TranslationTask):

     def __init__(self, args, src_dict, tgt_dict):
         super().__init__(args, src_dict, tgt_dict)
+        self.args = args
         self.langs = args.langs.split(",")
         for d in [src_dict, tgt_dict]:
             for l in self.langs:
```
[here](https://github.com/pytorch/fairseq/blob/c2e8904b6072d8eddab362ac50b324e374b5951d/fairseq/tasks/translation_from_pretrained_bart.py#L54), but I have no idea whether you still need it in your (newer) version of fairseq.

3. I stored everything in `data` folder and run my scripts from `solution` folder, so the hierarchy looks this way:
```
- data/
-- data/preprocessed_data/
-- data/data-bin/
- solution/
```

Assuming that you are at the `solution` folder, run:
```bash
mkdir ../data
mkdir ../data/preprocessed_data/
```

And download the mBART checkpoint:
```bash 
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz
tar -xzvf mbart.cc25.v2.tar.gz -C ../data
```

## Data Preprocessing
Everything here is written in the assumption that you are running it from the `solution` folder.
1. Specify the following environment variables:
```bash
SPM=<path to sentencepiece>
BPE_MODEL=../data/mbart.cc25.v2/sentence.bpe.model
DATA_DIR=../data/preprocessed_data
PREPROCESSED_DATA_DIR=../data/data-bin
DICT=../data/mbart.cc25.v2/dict.txt
```

2. Preprocess everything with sentencepiece:
```bash
${SPM} --model=${BPE_MODEL} < ${DATA_DIR}/train.src > ${DATA_DIR}/train.spm.src &
${SPM} --model=${BPE_MODEL} < ${DATA_DIR}/valid.src > ${DATA_DIR}/valid.spm.src &
${SPM} --model=${BPE_MODEL} < ${DATA_DIR}/train.dst > ${DATA_DIR}/train.spm.dst &
${SPM} --model=${BPE_MODEL} < ${DATA_DIR}/valid.dst > ${DATA_DIR}/valid.spm.dst &
```

3. Add the control tokens to the data:
```bash
python add_control_tokens.py
```
It finds the control tokens for each `(src, dst)` pair and finds unused tokens from the dictionary that can be used for the conditioning. It would have been cleaner to use some new tokens for this purpose, but I didn't know for sure how to add new tokens to a model in fairseq, so I stuck to a hackier option.

4. Run the binarization function:
```bash
fairseq-preprocess \
  --source-lang src \
  --target-lang dst \
  --trainpref ${DATA_DIR}/train.spm \
  --validpref ${DATA_DIR}/valid.spm \
  --destdir ${PREPROCESSED_DATA_DIR} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 70
```

5. Train:
```bash
PRETRAIN=../data/mbart.cc25.v2/model.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
CUDA_VISIBLE_DEVICES=0
fairseq-train ../data/data-bin \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 10000 --total-num-update 100000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 \
  --source-lang src --target-lang dst \
  --batch-size 8 \
  --update-freq 4 \
  --validate-interval 1 \
  --patience 3 \
  --max-epoch 25 \
  --save-interval-updates 500 --keep-interval-updates 1 --keep-best-checkpoints 1 --no-save-optimizer-state \
  --seed 42 --log-format tqdm \
  --restore-file ${PRETRAIN} \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --ddp-backend no_c10d \
  --langs $langs \
  --scoring bleu \
  --save-dir ../checkpoints
```
Find batch-size and update-freq that suits your gpu better, use fp16 whenever it's possible.

6. Generate:
```bash
CUDA_VISIBLE_DEVICES=0
LANG=C.UTF-8 LC_ALL=C.UTF-8
fairseq-generate ${DATA_DIR} \
  --path ${SAVE_DIR}/checkpoint_best.pt \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  --source-lang src --target-lang dst \
  --bpe 'sentencepiece' --sentencepiece-model ${BPE_MODEL} \
  --sacrebleu --remove-bpe 'sentencepiece' \
  --batch-size 32 --langs $langs > model_prediction.txt & 

cat model_prediction.txt | grep -P "^H" |sort -V |cut -f 3- > model_prediction.hyp
```

7. Find the control tokens that work better. Generate data with random control token combinations and choose the best by SARI on the dev set:
```bash
python generate_devs_with_control_tokens.py
```

I used the following params:
- NbChars=0.95;
- LevSim=0.4; 
- WordRank_1.6.

To be honest, the model hallucinate a lot in such setup, but SARI prefers it to any other (more sane in my opition) combination of tokens...
