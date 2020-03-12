# Using self trained model to dehazing. 

echo python3 MAINTHREAD_test.py \
    && python3 MAINTHREAD_test.py \
        --cuda \
        --normalize \
        --model ./pretrained-model/AtJ_DH_MODEL.pth \
        --test ./test_images/NTIRE2020_RAW/val/Hazy \
        --gt ./test_image/NTIRE2020_RAW/val/GT \
        --outdir ./test_images/NTIRE2020_RAW/val/DeHazy \
        --parse ./test_images/NTIRE2020_RAW/val/Parse \
        --rehaze ./test_images/NTIRE2020_RAW/val/ReHazy

echo python3 metric.py \
    && python3 metric.py \
        --GT_dir ./test_images/NTIRE2020_RAW/val/GT \
        --DH_dir ./test_images/NTIRE2020_RAW/val/DeHazy
