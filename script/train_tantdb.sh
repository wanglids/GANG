exp_name="baseline"
gpu=-1
ratio=1
resolution=-1
appearance_dim=0

fork=2
base_layer=10
visible_threshold=-1 #0.9 
dist2level="round"
update_ratio=0.2

progressive="True"
dist_ratio=0.999 #0.99
levels=-1
init_level=-1
extra_ratio=0.25
extra_up=0.01
# wo_sdf  #all wo_IIF wo_local
# modellists="wo_SG wo_normal"
# for modename in $modellists
# do
#     list="drjohnson playroom train truck" 
modellists="wo_IIF wo_local wo_normal wo_SG wo_TV wo_Env"
for modename in $modellists
do
    list="drjohnson truck" 

    for i in $list
    do
        iterations=25000
        pbr_iterations=40000
        source_path=/a800data1/lideqi/data/${i}
        model_path=outputs/tantdb/${i}_${modename}


        # python train.py --eval -s ${source_path} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} \
        # --iterations ${iterations} -m ${model_path} --appearance_dim ${appearance_dim} \
        # --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
        # --progressive --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
        #         --extra_ratio ${extra_ratio} --extra_up ${extra_up} --checkpoint_iterations ${iterations}
       
        checkpoint=${model_path}/chkpnt${iterations}.pth

  
        python train.py --eval -s ${source_path} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} \
        --iterations ${pbr_iterations} -m ${model_path} --appearance_dim ${appearance_dim} \
        --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
        --progressive --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
        --extra_ratio ${extra_ratio} --extra_up ${extra_up} --checkpoint_iterations ${pbr_iterations} --is_pbr --start_checkpoint ${checkpoint}

    done
done
