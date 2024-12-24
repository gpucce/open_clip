cd /raid/homes/giovanni.puccetti/Repos/dinoclip/open_clip/src
export CUDA_VISIBLE_DEVICES="0,6"
torchrun --nproc_per_node 2 --master-addr 127.0.0.1 \
    --master-port 19500 \
    -m open_clip_train.main \
    --train-data '/raid/datasets/laion400m-data/{00000..00654}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 4 \
    --precision amp \
    --model "ViT-L-16" \
    --workers 2 \
    --logs /raid/homes/giovanni.puccetti/Repos/dinoclip/logs \
    --dino-config-file /raid/homes/giovanni.puccetti/Repos/dinoclip/dinov2/dinov2/configs/train/vitl16_short.yaml \
    --dino-repo-path /raid/homes/giovanni.puccetti/Repos/dinoclip/dinov2/
