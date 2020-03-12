# Using self trained model to dehazing. 

echo python3 MAINTHREAD_test.py \
    && python3 MAINTHREAD_test.py \
        --cuda \
        --normalize \
        --model ./pretrained-model/AtJ_DH_MODEL.pth \
        --test ../dataset/DS4_2020/val/Hazy \
        --gt ../dataset/DS4_2020/val/GT \
        --outdir ./test_images/DS4_2020/val/DeHazy \
        --parse ./test_images/DS4_2020/val/Parse \
        --rehaze ./test_images/DS4_2020/val/ReHazy

echo python3 metric.py \
    && python3 metric.py \
        --GT_dir ../dataset/DS4_2020/val/GT \
        --DH_dir ./test_images/DS4_2020/val/DeHazy
