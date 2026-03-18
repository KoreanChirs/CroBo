accum_iter=8
batch_size=24
epochs=400
warmup_epochs=40
local_data_path=/mnt/tmp/kinetics400/videos
mask_ratio=0.9
model_name=vit_small_patch16
repeated_sampling=2
save_path=/mnt/tmp/checkpoints
crobo_path=sm
num_gpus_per_node=$(nvidia-smi -L | wc -l)

python -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node} main_pretrain_crobo.py \
     --batch_size ${batch_size} \
     --accum_iter ${accum_iter} \
     --model crobo_${model_name} \
     --epochs ${epochs} \
     --warmup_epochs ${warmup_epochs} \
     --data_path ${local_data_path} \
     --output_dir ${save_path}/output \
     --norm_pix_loss \
     --repeated_sampling ${repeated_sampling} \
     --mask_ratio ${mask_ratio} \
     --crobo_path ${crobo_path}
