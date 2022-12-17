metric="macro"
knowledge="U_and_F"
for seed in 42 0 1 2 3
do
  cmd="python train.py \
    --name base_UF_${metric}_$seed \
    --model BaseModel \
    --dataset daily_dialogue \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --seed $seed \
    --metric $metric \
    --scheduler fixed \
    --lr 0.00001 \
    --plm_lr 5e-6 \
    --feature_metric $metric \
    --epochs 30 \
    --knowledge $knowledge"
  echo "${cmd}"
  eval $cmd
done