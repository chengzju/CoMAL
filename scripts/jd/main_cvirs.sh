python ./main.py \
--data_dir ./data/jd \
--maxlength 64 \
--batch_size 64 \
--unlab_batch_size 64 \
--test_size 1500 \
--total_patience 6 \
--save_path ./model_saved/jd \
--lr 1e-4 \
--gpuid '1' \
--cycles 51 \
--try_id 1 \
--seed 1 \
--method_type cvirs \
--init_example_num 50 \
--well_init_lower_bound 1 \
--sample_pair_num 50 \
--dynamic_split \
--test_data_size 5000 \
--well_init \
--freeze_bert \
--freeze_layer_num 9 \