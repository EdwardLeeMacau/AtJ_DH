# Using self trained model to dehazing.

echo python3 MAINTHREAD_test.py \
    && python3 MAINTHREAD_test.py \
        --cuda \
        --model ./model/At_model2.pth \
        --test ./test_images/NTIRE2019_RAW/test/Hazy \
        --gt ./test_image/NTIRE2019_RAW/test/GT \
        --parse ./test_image/NTIRE2019_RAW/test/Parse \
        --outdir ./test_images/NTIRE2019_RAW/test/DeHazy \
        --rehaze ./test_images/NTIRE2019_RAW/test/ReHazy

echo python3 metric.py \
    && python3 metric.py \
        --GT_dir ./test_images/NTIRE2019_RAW/test/GT \
        --DH_dir ./test_images/NTIRE2019_RAW/test/DeHazy
