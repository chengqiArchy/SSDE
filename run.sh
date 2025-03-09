conda activate dev_co

python train_ssde.py --seed 110 --wandb_project_name ssde_result --default_beta 0.3 --is_store_everything --use_input_sensitive --reset_log_std