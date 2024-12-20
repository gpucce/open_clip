cd open_clip/src
export CUDA_VISIBLE_DEVICES="2,3,4,7"
torchrun --nproc_per_node 4 --master-addr 127.0.0.1 \
    --master-port 19500 \
    -m open_clip_train.main \
    --train-data '/raid/datasets/laion400m-data/{00000..00654}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4