# Using self trained model to dehazing.

echo python3 MY_MAINTHREAD_test_2.py \
    && python3 MY_MAINTHREAD_test_2.py \
        --netG ./pretrained-model/AtJ_DH_MaxCKPT.pth \
        --haze ./test_images/NTIRE2020_RAW/test/Hazy \
        --dehaze ./test_images/NTIRE2020_RAW/test/DeHazy \
        --gpus 2

echo cp ./test_images/NTIRE2020_RAW/test/DeHazy/* ./submission/ \
    && cp ./test_images/NTIRE2020_RAW/test/DeHazy/* ./submission/
