""" Using author provided model to dehazing. """

echo python3 dehazetest.py \
    && python3 dehazetest.py \
        --netG ./model/At_model2.pth \
        --valDataroot ./test_images/NTIRE2019_RAW/test/Hazy_Patch \
        --outdir ./test_images/NTIRE2019_RAW/test/DeHazy_Patch \
        --batchSize 1 \
        --valBatchSize 1

echo python3 merge_patch.py \
    && python3 merge_patch.py \
        --GT_dir ./test_images/NTIRE2019_RAW/test/GT \
        --inputDir ./test_images/NTIRE2019_RAW/test/DeHazy_Patch \
        --outputDir ./test_images/NTIRE2019_RAW/test/DeHazy

echo python3 metric.py \
    && python3 metric.py \
        --GT_dir ./test_images/NTIRE2019_RAW/test/GT \
        --DH_dir ./test_images/NTIRE2019_RAW/test/DeHazy
