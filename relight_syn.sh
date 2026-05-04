iterations=40000
source_path=/lideqi/data/Synthetic4Relight/
model_path=outputs/Synthetic4Relight/

checkpoint=${model_path}/chkpnt${iterations}.pth

python relight.py \
    -m ${model_path} \
    -s ${source_path} \
    --checkpoint ${checkpoint} \
    --eval \
    --gamma
