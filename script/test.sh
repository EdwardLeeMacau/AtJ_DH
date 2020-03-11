""" Using self trained model to dehazing. """

echo python3 MAINTHREAD_test.py \
    && python3 MAINTHREAD_test.py \
        --cuda \
        --model ./pretrained_model/AtJ_DH_MaxCKPT.pth \
        --test ./test_images/NTIRE2020_RAW/test/Hazy \
        --outdir ./test_images/NTIRE2020_RAW/test/DeHazy

