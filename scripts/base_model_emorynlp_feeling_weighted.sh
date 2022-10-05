metric="weighted"
knowledge="feeling"
for seed in 42 0 1 2 3
do
  cmd="python train.py \
    --name base_emorynlp_feeling_${metric}_$seed \
    --model BaseModel \
    --dataset emorynlp \
    --batch_size 6 \
    --gradient_accumulation_steps 4 \
    --seed $seed \
    --metric $metric \
    --scheduler cosine \
    --lr 0.001 \
    --plm_lr 3e-6 \
    --n_sentences 3 \
    --feature_metric weighted \
    --epochs 20 \
    --knowledge $knowledge"
  echo "${cmd}"
  eval $cmd
done