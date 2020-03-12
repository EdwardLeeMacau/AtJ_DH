# Using self trained model to dehazing.

echo python3 MAINTHREAD_test.py \
    && python3 MAINTHREAD_test.py \
        --cuda \
        --normalize \
        --model ./pretrained-model/AtJ_DH_MODEL.pth \
        --test ./test_images/NTIRE2020_RAW/test/Hazy \
        --parse ./test_images/NTIRE2020_RAW/test/Parse \
        --outdir ./test_images/NTIRE2020_RAW/test/DeHazy

echo cp ./test_images/NTIRE2020_RAW/test/DeHazy/* ./submission/ \
    && cp ./test_images/NTIRE2020_RAW/test/DeHazy/* ./submission/
