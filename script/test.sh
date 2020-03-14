# Using self trained model to dehazing.

echo python3 MY_MAINTHREAD_test.py \
    && python3 MY_MAINTHREAD_test.py \
        --netG ../backup0312/AtJ_DH_MaxCKPT.pth \
        --haze ./test_images/NTIRE2020_RAW/test/Hazy \
        --dehaze ./test_images/NTIRE2020_RAW/test/DeHazy

echo cp ./test_images/NTIRE2020_RAW/test/DeHazy/* ./submission/ \
    && cp ./test_images/NTIRE2020_RAW/test/DeHazy/* ./submission/
