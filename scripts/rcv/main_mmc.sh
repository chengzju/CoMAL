python ./main.py \
--data_dir ./data/rcv1-v2 \
--maxlength 256 \
--batch_size 64 \
--unlab_batch_size 64 \
--train_size 23149 \
--test_size 10000 \
--total_patience 6 \
--save_path ./model_saved/rcv1 \
--lr 1e-4 \
--gpuid '0' \
--cycles 51 \
--try_id 1 \
--seed 1 \
--method_type mmc \
--init_example_num 200 \
--well_init_lower_bound 1 \
--sample_pair_num 100 \
--dynamic_split \
--test_data_size 5000 \
--well_init \
--freeze_bert \
--freeze_layer_num 9 \
