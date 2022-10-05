metric="macro"
knowledge="comet"
for seed in 42 0 1 2 3
do
  cmd="python train.py \
    --name base_emorynlp_comet_${metric}_$seed \
    --model BaseModel \
    --dataset emorynlp \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --seed $seed \
    --metric $metric \
    --scheduler cosine \
    --lr 0.001 \
    --plm_lr 2e-6 \
    --n_sentences 2 \
    --feature_metric macro \
    --epochs 30 \
    --knowledge $knowledge"
  echo "${cmd}"
  eval $cmd
done