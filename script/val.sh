# Using self trained model to dehazing. 

echo python3 MY_MAINTHREAD_test.py \
    && python3 MY_MAINTHREAD_test.py \
        --netG ./pretrained-model/AtJ_DH_CKPT.pth \
        --haze ./test_images/DS5_2020/val512/Hazy \
        --dehaze ./test_images/DS5_2020/val512/DeHazy

echo python3 metric.py \
    && python3 metric.py \
        --GT_dir ./test_images/DS5_2020/val512/GT \
        --DH_dir ./test_images/DS5_2020/val512/DeHazy
