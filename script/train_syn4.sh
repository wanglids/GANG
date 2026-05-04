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
# air_baloons chair hotdog jugs
# list="air_baloons jugs" #正常训练
list="air_baloons jugs chair hotdog"

for i in $list
do
    modellists="new"
    for modename in $modellists
    do
        iterations=25000 
        pbr_iterations=40000
        source_path=/lideqi/data/Synthetic4Relight/${i}
        model_path=outputs/Synthetic4Relight/${i}_${modename}

        if [ "$progressive" = "True" ]; then
            if [ "$warmup" = "True" ]; then
                python train.py --eval -s ${source_path} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} --warmup \
                --iterations ${iterations} -m ${model_path} --appearance_dim ${appearance_dim} \
                --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
                --progressive --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
                --extra_ratio ${extra_ratio} --extra_up ${extra_up} --checkpoint_iterations ${iterations} --normal_detal
            else
                python train.py --eval -s ${source_path} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} \
                --iterations ${iterations} -m ${model_path} --appearance_dim ${appearance_dim} \
                --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
                --progressive --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
                --extra_ratio ${extra_ratio} --extra_up ${extra_up} --checkpoint_iterations ${iterations} --normal_detal
            fi
        else
            if [ "$warmup" = "True" ]; then
                python train.py --eval -s ${source_path} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} --warmup \
                --iterations ${iterations} -m ${model_path} --appearance_dim ${appearance_dim} \
                --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
                --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
                --extra_ratio ${extra_ratio} --extra_up ${extra_up} --checkpoint_iterations ${iterations} --normal_detal
            else
                python train.py --eval -s ${source_path} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} \
                --iterations ${iterations} -m ${model_path} --appearance_dim ${appearance_dim} \
                --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
                --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
                --extra_ratio ${extra_ratio} --extra_up ${extra_up} --checkpoint_iterations ${iterations} --normal_detal
            fi
        fi

        # python baking_occlu.py \
        #     -m ${model_path} \
        #     --checkpoint ${model_path}/chkpnt${iterations}.pth \
        #     --bound 1.5 \
        #     --occlu_res 128 \
        #     --occlusion 0.25
            
        
        checkpoint=${model_path}/chkpnt${iterations}.pth

        if [ "$progressive" = "True" ]; then
            if [ "$warmup" = "True" ]; then
                python train.py --eval -s ${source_path} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} --warmup \
                --iterations ${pbr_iterations} -m ${model_path} --appearance_dim ${appearance_dim} \
                --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
                --progressive --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
                --extra_ratio ${extra_ratio} --extra_up ${extra_up} --checkpoint_iterations ${pbr_iterations} --is_pbr --start_checkpoint ${checkpoint} --normal_detal
            else
                python train.py --eval -s ${source_path} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} \
                --iterations ${pbr_iterations} -m ${model_path} --appearance_dim ${appearance_dim} \
                --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
                --progressive --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
                --extra_ratio ${extra_ratio} --extra_up ${extra_up} --checkpoint_iterations ${pbr_iterations} --is_pbr --start_checkpoint ${checkpoint} --normal_detal
            fi
        else
            if [ "$warmup" = "True" ]; then
                python train.py --eval -s ${source_path} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} --warmup \
                --iterations ${pbr_iterations} -m ${model_path} --appearance_dim ${appearance_dim} \
                --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
                --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
                --extra_ratio ${extra_ratio} --extra_up ${extra_up} --checkpoint_iterations ${pbr_iterations} --is_pbr --start_checkpoint ${checkpoint} --normal_detal
            else
                python train.py --eval -s ${source_path} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} \
                --iterations ${pbr_iterations} -m ${model_path} --appearance_dim ${appearance_dim} \
                --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
                --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
                --extra_ratio ${extra_ratio} --extra_up ${extra_up} --checkpoint_iterations ${pbr_iterations} --is_pbr --start_checkpoint ${checkpoint} --normal_detal
            fi
        fi
    done
done