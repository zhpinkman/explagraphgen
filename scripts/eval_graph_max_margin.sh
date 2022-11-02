python src/train_graph_gen_max_margin.py \
--data_dir=contrastive_data \
--model_name_or_path=models/max-margin/best_tfmr \
--learning_rate=3e-5 \
--train_batch_size=8 \
--eval_batch_size=8 \
--max_source_length=150 \
--max_target_length=150 \
--val_max_target_length=150 \
--test_max_target_length=150 \
--output_dir=models/max-margin \
--num_train_epochs=15  \
--cache_dir cache \
--gpus=1 \
--do_predict \
--eval_beams 4