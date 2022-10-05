metric="macro"
knowledge="none"
for seed in 42 0 1 2 3
do
  cmd="python train.py \
    --name infer_dd_${metric}_$seed \
    --model DialogueInfer \
    --dataset daily_dialogue \
    --batch_size 16 \
    --lr 0.0005 \
    --scheduler fixed \
    --gradient_accumulation_steps 1 \
    --seed $seed \
    --metric $metric \
    --epochs 60"
  echo "${cmd}"
  eval $cmd
done