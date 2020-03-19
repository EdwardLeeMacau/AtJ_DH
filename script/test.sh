# Using self trained model to dehazing.

# echo python3 script/resize_crop_shift.py \
#     && python3 script/resize_crop_shift.py \
#         --inputDir ./test_images/NTIRE2020_RAW/test/Hazy \
#         --outputDir ./test_images/NTIRE2020_RAW/test/Hazy_Patch \
#         --resize 1024 \
#         --segment 5        

# echo python3 MY_MAINTHREAD_test_2.py \
#     && python3 MY_MAINTHREAD_test_2.py \
#         --netG ./pretrained-model/AtJ_DH_MaxCKPT.pth \
#         --haze ./test_images/NTIRE2020_RAW/test/Hazy_Patch \
#         --dehaze ./test_images/NTIRE2020_RAW/test/DeHazy_Patch \
#         --gpus 2

echo python3 merge_patch_spline.py \
    && python3 merge_patch_spline.py \
        --ref ./test_images/NTIRE2020_RAW/test/Hazy \
        --patch ./test_images/NTIRE2020_RAW/test/DeHazy_Patch \
        --merge ./test_images/NTIRE2020_RAW/test/DeHazy \
        --segment 5 \
        --L 1200

echo cp ./test_images/NTIRE2020_RAW/test/DeHazy/* ./submission/ \
    && cp ./test_images/NTIRE2020_RAW/test/DeHazy/* ./submission/
