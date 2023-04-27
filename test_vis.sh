NUM_GPU=1
# bash video_tracking/tools/dist_test.sh $1 $NUM_GPU --checkpoint $2 --cfg-options model.track_head.pixel_decoder.encoder.layer_cfg.self_attn_cfg.im2col_step=1
# bash video_tracking/tools/dist_test.sh video_tracking/configs/vis/mask2former/$1.py $NUM_GPU --checkpoint $2 --cfg-options model.track_head.pixel_decoder.encoder.layer_cfg.self_attn_cfg.im2col_step=1 val_dataloader.outfile_prefix=$1 test_evaluator.outfile_prefix=$1
bash video_tracking/tools/dist_test.sh configs/mask2former/$1.py $NUM_GPU --checkpoint $2 --cfg-options model.track_head.pixel_decoder.encoder.layer_cfg.self_attn_cfg.im2col_step=1 val_dataloader.outfile_prefix=$1 test_evaluator.outfile_prefix=$1

