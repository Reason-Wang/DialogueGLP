metric="weighted"
feature_metric="weighted"
for seed in 42 0 1 2 3
do
    cmd="python train.py \
      --name crn_meld_${metric}_$seed \
      --model DialogueCRN \
      --dataset meld \
      --batch_size 32 \
      --scheduler fixed \
      --gradient_accumulation_steps 1 \
      --lr 0.0005 \
      --seed $seed \
      --metric $metric \
      --feature_metric $feature_metric \
      --epochs 40"
  echo "${cmd}"
  eval $cmd
done