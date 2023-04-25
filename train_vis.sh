N_ARGS=$#
for EXP in $* ;  
do  
CMD="bash video_tracking/tools/dist_train.sh $EXP 8"
echo $CMD;  
eval $CMD;
done  
