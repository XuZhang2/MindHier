export HF_ENDPOINT=https://hf-mirror.com 
export NCCL_CROSS_NIC=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
python train.py \
  --vit_version='VIT_L' \
  --data_path data/nsd \
  --images_dir data/images \
  --captions_path data/annotations/COCO_73k_annots_curated.npy \
  --output_dir outputs/brainencoder
# accelerate launch --multi_gpu --num_processes 1 --gpu_ids 0 --main_process_port 29503 train.py --vit_version='VIT_L' --subj=1 --num_epochs=300 --batch_size=128\

# accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 --main_process_port 29502 train.py \
#                  --fmri_encoder 'brainxs' \


# accelerate launch  --num_processes 1 --gpu_ids 0 --main_process_port 29502 train.py \
#                  --fmri_encoder 'brainxs' \

# accelerate launch  --num_processes 1 --gpu_ids 3 --main_process_port 29502 train_multi_thread.py \
#                  --fmri_encoder 'brainxs' \0
