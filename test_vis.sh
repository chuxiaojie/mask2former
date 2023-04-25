NUM_GPU=8
bash video_tracking/tools/dist_test.sh $1 $NUM_GPU --checkpoint $2 --cfg-options model.track_head.pixel_decoder.encoder.layer_cfg.self_attn_cfg.im2col_step=1