metric="macro"
knowledge="none"
feature_metric="macro"
for seed in 42 0 1 2 3
do
  cmd="python train.py \
    --name infer_emorynlp_${metric}_$seed \
    --model DialogueInfer \
    --dataset emorynlp \
    --batch_size 16 \
    --scheduler fixed \
    --lr 0.0003 \
    --gradient_accumulation_steps 1 \
    --seed $seed \
    --metric $metric \
    --feature_metric $feature_metric \
    --epochs 60"
  echo "${cmd}"
  eval $cmd
done