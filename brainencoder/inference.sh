subj_test=$1
text_image_ratio=0.5
guidance=5
gpu_id=$2
retrival_strength=0.1
timestamp="Apr05_05-10-45"
model_name="baseline_10w_600e_8e_5_bs256_crop_one_region_8e_5_last_5_test"
data_path="data/nsd"                    # TODO: replace with your NSD root
train_logs_dir="outputs/brainencoder"   # TODO: replace with your Stage 1 checkpoint root
recon_output_dir="outputs/recon"         # TODO: replace with your reconstruction output root

CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
recon.py \
--subj_test $subj_test \
--data_path $data_path \
--text_image_ratio $text_image_ratio --guidance $guidance \
--recons_per_sample 10 \
--model_name $model_name \
--train_logs_dir $train_logs_dir \
--recon_output_dir $recon_output_dir \
--retrival_from_memory_strength $retrival_strength \
--rec_timestamp $timestamp \
--no-plotting \
--no-retrival_from_memory
# --test_end 2 \
# --test_start 628 \
# --response_path $response_path \

# results_path="./recon/"$model_name"/recon_on_subj"$subj_test

# CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
# eval.py --results_path $results_path
