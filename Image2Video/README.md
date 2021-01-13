### Training

```
python train.py  \
      --image_batch 32 \
      --video_batch 32 \
      --use_categories \
      --use_infogan \
      --use_noise \
      --noise_sigma 0.1 \
      --image_discriminator PatchImageDiscriminator \
      --video_discriminator CategoricalVideoDiscriminator \
      --print_every 1 \
      --every_nth 2 \
      --batches 100000 \
      --dim_z_content 50 \
      --dim_z_motion 10 \
      --dim_z_category 3 \
      --video_length 16  \
     ./data_test \
     ./logs/tobigsfin/
```
### Inference

```
python generate_videos.py \ 
    ./logs/fin/generator_10000.pytorch \        # model path
    ./infer/img_0000.jpg \                      # input image path
    disgust \                                   # class : "disgust" or "surprise" or "happiness" 
    ./                                          # save dir path 
```
