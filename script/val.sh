# Using self trained model to dehazing. 

echo python3 MY_MAINTHREAD_test.py \
    && python3 MY_MAINTHREAD_test.py \
        --netG ../backup0312/AtJ_DH_MaxCKPT.pth \
        --haze ./test_images/DS4_2020/val/Hazy \
        --dehaze ./test_images/DS4_2020/val/DeHazy

echo python3 metric.py \
    && python3 metric.py \
        --GT_dir ./test_images/DS4_2020/val/GT \
        --DH_dir ./test_images/DS4_2020/val/DeHazy
