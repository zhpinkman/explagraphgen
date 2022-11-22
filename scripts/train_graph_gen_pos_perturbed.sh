python src/train_graph_gen_pos_perturbed.py \
--data_dir=contrastive_data \
--model_name_or_path=t5-large \
--learning_rate=3e-5 \
--train_batch_size=8 \
--eval_batch_size=8 \
--max_source_length=150 \
--max_target_length=150 \
--val_max_target_length=150 \
--test_max_target_length=150 \
--output_dir=models/pos_aug \
--overwrite_output_dir \
--num_train_epochs=5  \
--cache_dir cache \
--logger_name wandb \
--gpus=1 \
--do_train \
--do_predict \
--eval_beams 4