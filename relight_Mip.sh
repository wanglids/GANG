
iterations = 40000
source_path=/home/dqli/program/data/Mip-NeRF360/bicycle
model_path=outputs/Mip-NeRF360/bicycle_new
checkpoint=${model_path}/chkpnt${iterations}.pth

python relight.py \
    -m ${model_path} \
    -s ${source_path} \
    --checkpoint ${checkpoint} \
    --eval \
    --gamma



