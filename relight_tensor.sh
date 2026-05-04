# iterations=40000

# source_path=/lideqi/data/TensoIRSynthetic/
# model_path=outputs/TensoIRSynthetic/

# checkpoint=${model_path}/chkpnt${iterations}.pth

# python relight_ab.py \
#     -m ${model_path} \
#     -s ${source_path} \
#     --checkpoint ${checkpoint} \
#     --eval \
#     --gamma


list="lego ficus armadillo hotdog" 


# for i in $list
# do
#     source_path=/lideqi/data/TensoIRSynthetic/${i}
#     model_path=outputs/TensoIRSynthetic_new/${i}

#     checkpoint=${model_path}/chkpnt${iterations}.pth

#     python relight.py \
#         -m ${model_path} \
#         -s ${source_path} \
#         --checkpoint ${checkpoint} \
#         --eval \
#         --gamma
# done

# list="lego" 
source_path=/lideqi/data/TensoIRSynthetic/
model_path=outputs/TensoIRSynthetic/lego_new

checkpoint=${model_path}/chkpnt${iterations}.pth

python relight_ab.py \
    -m ${model_path} \
    -s ${source_path} \
    --checkpoint ${checkpoint} \
    --eval \
    --gamma

