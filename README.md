# 🙂Moving-Emoji-Generation🤥
<img src="https://user-images.githubusercontent.com/41895063/104711870-3eb08000-5765-11eb-9223-76b7db582395.png" width="700" height="400">


## Tobigticon - Create Your Own Moving Emoticons Based on Image2Video.

Tobigticon is a service that provides the production of your own moving emoticon using GAN based image2video method.  

  
<img src="https://user-images.githubusercontent.com/68625698/104740511-61558f80-578b-11eb-8772-6b2a7105d1e9.png" width="700" height="500">  

## Notice

EmoGE'T, which is derived from Emoji GEnerated by Tobigs, is a team of 8 members of Tobigs who worked on an image2video based project.

Following the below options, the trained model creates your own animated emoji.


**Select an Image Style**
| Disney  |  Baby | Painting  | 
|---|---|---|
|  <img src="images/anime.jpg" width="150" height="150"> |  <img src="images/baby.jpeg" width="150" height="150"> |  <img src="images/painting.jpeg" width="150" height="150"> | 


**Select Emoji Emotion**
| Happiness  |  Disgusted | Surprise  | 
|---|---|---|
|  <img src="images/happy.png" width="150" height="150"> |  <img src="images/disgust.png" width="150" height="150"> |  <img src="images/surprise.png" width="150" height="150"> | 

The Web page demonstrated at the 11th conference of Tobigs is Demo web page[].  
Make your own moving emoticons with Tobigticons! 

## Requirements

We have tested on:

- CUDA 11.0
- python 3.8.5
- pytorch 1.7.1
- numpy 1.19.2
- opencv-python  4.5.1
- dlib 19.21.1
- scikit-learn 0.24.0
- Pillow 8.1.0
- Ninja 1.10.0
- glob2 0.7

## Usage

### Generate your own Emoji

You can generate your own moving emoticon :)
```
python emoticon_generate.py --file ImagePath --transform Animation --emotion Emotion --type OutputType --model Approach 
```

For example,
``` 
python emoticon_generate.py --file 00001.jpg --transform baby --emotion disgusted --type mp4 --model sol1
```

### Training

Training the landmark generation model using the Sol1 approach that developed WARP.

``` 
python sol1/main.py --data_path DataPath --conditions Conditions
```

Train the video generation model using sol2 developed by mocogan.

```
python sol2/train.py --image_discriminator PatchImageDiscriminator --video_discriminator CategoricalVideoDiscriminator --dim_z_category 3 --video_length 16  
```

Generate the video using sol2 model

``` 
python sol2/generate_videos.py [model path] [image] [class] [save_path]
```

## Pretrained Checkpoints

[Animation](https://drive.google.com/file/d/1JO5-MJwjzrCYCVL-2StAKJbGy-VuwBSp/view?usp=sharing)
[Baby](https://drive.google.com/file/d/1Oplgd05plKLa76QzmFE8GsghpZOiMia-/view?usp=sharing)
[Painting](https://drive.google.com/file/d/1I0BpprKG3hXixQsnzOezTaG8SZr97OkB/view?usp=sharing)

## Samples
  
### Disney - Happy
|Original|Generative|
|---|---|
|<img src="images/anime-happy.png" width="300" height="300">|<img src="https://github.com/minjung-s/Moving-Emoji-Generation/blob/main/images/anime-happy.gif" width="300" height="300">|  
  
###  Baby - Happy
|Original|Generative|
|---|---|
|<img src="images/baby-happy.png" width="300" height="300">|<img src="https://github.com/minjung-s/Moving-Emoji-Generation/blob/main/images/baby-happy.gif" width="300" height="300">|  
  
### Painting - Happy  
|Original|Generative|
|---|---|
|<img src="images/painting-happy.png" width="300" height="300">|<img src="https://github.com/minjung-s/Moving-Emoji-Generation/blob/main/images/painting-happy.gif" width="300" height="300">|   
  
### Disney - Surprise  
|Original|Generative|
|---|---|
|<img src="images/anime-surprise.png" width="300" height="300">|<img src="https://github.com/minjung-s/Moving-Emoji-Generation/blob/main/images/anime-surprise.gif" width="300" height="300">|  
  
### Baby - Surprise
|Original|Generative|
|---|---|
|<img src="images/baby-surprise.png" width="300" height="300">|<img src="https://github.com/minjung-s/Moving-Emoji-Generation/blob/main/images/baby-surprise.gif" width="300" height="300">|    
  
### Painting - Surprise  
|Original|Generative|
|---|---|
|<img src="images/painting-surprise.png" width="300" height="300">|<img src="https://github.com/minjung-s/Moving-Emoji-Generation/blob/main/images/painting-surprise.gif" width="300" height="300">|  

## Reference
- Rosinality, stylegan2-pytorch,  2019, [https://github.com/rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)
- PieraRiccio, stylegan2-pytorch, 2019, [https://github.com/PieraRiccio/stylegan2-pytorch](https://github.com/PieraRiccio/stylegan2-pytorch)
- justinpinkney, toonify, 2020, [https://github.com/justinpinkney/toonify](https://github.com/justinpinkney/toonify)
- marsbroshok, face-replace, 2016, https://github.com/marsbroshok/face-replace
- sergeytulyakov, mocogan, 2017, https://github.com/sergeytulyakov/mocogan
- Yaohui Wang, Piotr Bilinski, Francois Bremond, Antitza Dantcheva. ImaGINator: Conditional Spatio-Temporal GAN for Video Generation. 2019.


## Contributor 🌟
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<table>
  <tr>
    <td align="center"><a href="https://github.com/minjung-s"><img src="https://user-images.githubusercontent.com/41895063/104710310-45d68e80-5763-11eb-8bca-3d11bab3f636.png" width="150" height="150"><br /><sub><b>MinJung Shin</b></sub></td>
    <td align="center"><a href="https://github.com/simba-pumba"><img src="https://user-images.githubusercontent.com/41895063/104711155-576c6600-5764-11eb-8a1e-a7a011cfe17d.png" width="150" height="150"><br /><sub><b>YeJi Lee</b></sub></td>
    <td align="center"><a href="https://github.com/yourmean"><img src="https://user-images.githubusercontent.com/41895063/104711276-7cf96f80-5764-11eb-8473-99c5c0dc8a8a.png" width="150" height="150"><br /><sub><b>YuMin Lee</b></sub></td>
    <td align="center"><a href="https://github.com/lilly9117"><img src="https://user-images.githubusercontent.com/41895063/104711018-29872180-5764-11eb-9858-53c5f4cc26e4.png" width="150" height="150"><br /><sub><b>Hyebin Choi</b></sub></td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><a href="https://github.com/minkyeong"><img src="https://user-images.githubusercontent.com/55529646/104719170-6a386800-576f-11eb-8b55-36b524c4d166.jpg" width="150" height="150"><br /><sub><b>MinKyeong Kim</b></sub></td>
    <td align="center"><a href="https://github.com/shkim960520"><img src="https://user-images.githubusercontent.com/55529646/104719176-6d335880-576f-11eb-849f-4d6756824d68.jpg" width="150" height="150"><br /><sub><b>SangHyun Kim</b></sub></td>
    <td align="center"><a href="https://github.com/Jeong-JaeYoon"><img src="https://user-images.githubusercontent.com/55529646/104719173-6b699500-576f-11eb-9a42-a8569be057bc.jpg" width="150" height="150"><br /><sub><b>JaeYoon Jeong</b></sub></td>
    <td align="center"><a href="https://github.com/Yu-Jin22"><img src="https://user-images.githubusercontent.com/41895063/104711416-afa36800-5764-11eb-85c1-1a9ad50033b7.png" width="150" height="150"><br /><sub><b>YuJin Han</b></sub></td>
  </tr>
</table>
