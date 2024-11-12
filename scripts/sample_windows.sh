# windows
python main.py  --backend nccl --world_size 1 --local_gpus 1 --global_rank 0 \
                --wandb --project toy_model --run_name test \
                --output_dir toy_model\\test --auto_resume \
                --model toy_model --epoch 1 \
                --opt adamw --opt_lr 1e-4 --weight_decay 1e-4 --warmup_lr 1e-6 --min_lr 1e-6 --warmup_epochs 5 --T 50\
                --loss l1loss --loss_scaler \
                --cfg_dataset /research/Networks/torch_template/datasets/configs/sample_datasets.yaml \
                --save_ckpt --save_ckpt_freq 5 \
                --test