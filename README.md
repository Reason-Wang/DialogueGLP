## DialogueGLP: Global-Local Modeling with Prompt-Based Knowledge Enhancement for Emotion Inference in Conversation

This is code for "Global-Local Modeling with Prompt-Based Knowledge Enhancement for Emotion Inference in Conversation".

### Dependencies

```
torch = 1.10.1
torch_geometric = 2.0.4
transformers = 4.17.0
```

### Training

#### Training feature extractor

For most baselines and DialogueGLP, first train a feature extractor which is trained directly with (utterance, emotion) pairs. Take DailyDialog dataset as an example, the following trains a roberta-large feature extractor on DailyDialog.

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

#### Extracting features

After training with (utterance, emotion) pairs, we use the trained model from previous step to extractor features. 

```bash
python feature_extraction.py \
 --name feature_extractor_dd_weighted \
 --model Extractor \
 --dataset daily_dialogue \
 --metric weighted
```

#### Training DialogueGLP

After extracting the features, we can now train DialogueGLP. The following gives a shell script example to train DialogueGLP on DailyDialog with 5 random seeds. You can also find other scripts to train other models in *scripts* folder. 

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

### Testing New Models

You can also add a new model and test its performance of inferring emotions. Basically you should follow the following steps:

1. Add the model file to *model* folder and write a wrapper for the model in *model/base_models.py*
2. Add dataset and collator for that model in *data/dataset.py*
3. Add arguments in *utils/options.py*

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