## Global-Local Modeling with Prompt-Based Knowledge Enhancement for Emotion Inference in Conversation

### Dependencies

```
torch = 1.10.1
torch = 2.0.4
transformers = 4.17.0
```

### Training feature extractor

```bash
# DailyDialog weighted
python feature_tuning.py \
  --name feature_extractor_dd_weighted \
  --model Extractor \
  --dataset daily_dialogue \
  --batch_size 16 \
  --lr 5e-6 \
  --epochs 4 \
  --seed 42 \
  --metric weighted
```

### Extracting features

```bash
python feature_extraction.py \
 --name feature_extractor_dd_weighted \
 --model Extractor \
 --dataset daily_dialogue \
 --metric weighted
```

### Training DialogueGLP

```bash
# you can also run scripts/base_model_dd_weighted.sh
metric="weighted"
knowledge="none"
for seed in 42 0 1 2 3
do
  cmd="python train.py \
    --name base_${metric}_$seed \
    --model BaseModel \
    --dataset daily_dialogue \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --seed $seed \
    --metric $metric \
    --scheduler cosine \
    --lr 0.001 \
    --plm_lr 1e-5 \
    --feature_metric weighted \
    --epochs 16 \
    --knowledge $knowledge"
  echo "${cmd}"
  eval $cmd
done
```



### Directory Tree

```
DialogueGLP
├── data
│   ├── daily_dialogue
│   │   ├── knowledge
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── validation.csv
│   ├── emorynlp
│   │   ├── knowledge
│   │   ├── ...
│   │
│   ├── meld
│   │   ├── knowledge
│   │   ├── ...
│   │
│   ├── dataset.py
│   ├── load_data.py
│   └── process_data.py
├── model
│   ├── cog_bart
│   ├── com_pm
│   ├── dag_erc
│   ├── dialogue_gcn
│   ├── base_models.py
│   ├── dialogue_crn.py
│   ├── dialogue_infer.py
│   ├── dialogue_rnn.py
│   └── ei_roberta.py
├── scripts
│   ├── base_model_dd_weighted.sh
│   ├── ...
│   
├── utils
│   ├── options.py
│   ├── utils.py
│   ├── ...
│   
├── feature_extraction.py
├── feature_tuning.py
└── train.py
```

