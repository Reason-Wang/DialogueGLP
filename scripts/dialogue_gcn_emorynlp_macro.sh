metric="macro"
knowledge="none"
feature_metric="macro"
for seed in 42 0 1 2 3
do
  cmd="python train.py \
    --name gcn_emorynlp_${metric}_$seed \
    --model DialogueGCN \
    --dataset emorynlp \
    --batch_size 32 \
    --scheduler fixed \
    --lr 0.0001 \
    --gradient_accumulation_steps 1 \
    --seed $seed \
    --metric $metric \
    --feature_metric $feature_metric \
    --epochs 60"
  echo "${cmd}"
  eval $cmd
done