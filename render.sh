

#         source_path=/home/dqli/program/data/Mip-NeRF360/${i}
        # model_path=outputs/Mip-NeRF360/${i}_${modename}/
list="bicycle" 
for i in $list
do
    python render.py -s /home/dqli/program/data/Mip-NeRF360/${i} -m outputs/Mip-NeRF360/${i}_new \
        --checkpoint outputs/Mip-NeRF360/${i}_new/chkpnt40000.pth --iteration 40000 --is_pbr -r 4
done

