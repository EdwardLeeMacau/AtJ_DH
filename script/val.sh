""" Using self trained model to dehazing. """

echo python3 MAINTHREAD_test.py \
    && python3 MAINTHREAD_test.py \
        --model ./pretrained_model/AtJ_DH_MaxCKPT.pth \
        --test ./test_images/DS4_2020/test/Hazy \
        --gt ./test_image/DS4_2020/test/GT \
        --outdir ./test_images/DS4_2020/test/DeHazy \
        --rehaze ./test_images/DS4_2020/test/ReHazy

echo python3 metric.py \
    && python3 metric.py \
        --GT_dir ./test_images/DS4_2020/test/GT \
        --DH_dir ./test_images/DS4_2020/test/DeHazy
