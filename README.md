# DiffuseMix : Label-Preserving Data Augmentation with Diffusion Models (CVPR'2024)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://diffusemix.github.io/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2405.14881)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)]([https://www.linkedin.com/in/khawarislam/](https://www.youtube.com/watch?v=FcM4wgieDmU))
[![code](https://img.shields.io/badge/-Demo-red)](https://github.com/khawar-islam/diffuseMix)
[![Page Views Count](https://badges.toozhao.com/badges/01HRR1Z1PZQZ9PCVJ7MN2Q67HN/blue.svg)](https://badges.toozhao.com/stats/01HRR1Z1PZQZ9PCVJ7MN2Q67HN "Get your own page views count badge on badges.toozhao.com")


<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="DiffusMix">
</p>

<p align="center">
    <img src="images/diffuseMix_flower102.png" alt="DiffusMix Treasure">
</p>

---

### 📢 Latest Updates
- **Mar-11-24**: DiffuseMix paper is released [arxiv link](https://arxiv.org/abs/2405.14881). 

### 🚀 Getting Started
Setup anaconda environment using `environment.yml` file.

```
conda env create --name DiffuseMix --file=environment.yml
conda remove -n DiffuseMix --all # In case environment installation faileds
```

### 📝 List of Prompts 
Below is the list of prompts, other_prompts are the new prompts we tested.

```
prompts = ["Autumn", "snowy", "watercolor art","sunset", "rainbow", "aurora",
               "mosaic", "ukiyo-e", "a sketch with crayon"]

other_prompts= ["Pointillism", "sepia", "etching", "twilight", "grainy film texture",
            "spring bloom", "stormy sky", "pixel art", "golden hour"]          
```

### 📁 Dataset Structure
We used the FGVC (fine-grained visual classification) Aircraft Benchmark for model training.
(https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
```
aircraft_100cls
train
 └─── class 1
          └───── 0056978.jpg
 └─── class 2
          └───── 0054367.jpg
 └─── ...
```

### 🎶 Examples of Result
```
data_example
diffuseMix or diffuseMix_bbox
 └─── original
          └───── 0743290_dmix_0.jpg
 └─── other_prompts
          └───── 0817494_dmix_0.jpg
 └─── ...
```
<p align="center">
    <img src="data_example/bbox_vis/0817494.jpg" alt="Bounding Box Visualization">
</p>

<p align="center">
    <img src="data_example/diffuseMix_bbox/other_prompts/0817494_dmix_0.jpg" alt="Bounding Box Visualization">
</p>

### ✨ DiffuseMix Augmentation
To introduce the structural complexity, you can download fractal image dataset from here [Fractal Dataset](https://drive.google.com/drive/folders/1uxK7JaO1NaJxaAGViQa1bZfX6ZzNMzx2?usp=sharing)
IMAGES_BOX.TXT containts bounding box(bbox) coordinates for the single object in each image.
```
python3 main_prompt.py --train_dir PATH --fractal_dir PATH --out_dir PATH --device 0
python3 main_bbox.py --train_dir PATH --fractal_dir PATH --out_dir PATH --device 0 --bbox_file IMAGES_BOX.TXT PATH
```

### 💡 Train with augmented data
- `--data-root`: Path to your original training dataset (the existing dataset you normally train on).
  It should contain the standard split folders, e.g. `train/`, `val/`, and `test/`.

- `--dmix-root`: Path to the generated DiffuseMix dataset you created.
  This directory should be organized in `ImageFolder` format (one subfolder per class).

Example:
```bash
python3 train_resnet.py \
  --data-root /path/to/original_dataset \
  --dmix-root /path/to/generated_dmix_dataset \
  --save-dir /path/to/checkpoints \
```

### 💬 Citation
```
@article{diffuseMix2024,
  title={DIFFUSEMIX: Label-Preserving Data Augmentation with Diffusion Models},
  author={Khawar Islam, Muhammad Zaigham Zaheer, Arif Mahmood, Karthik Nandakumar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@techreport{maji13fine-grained,
   title         = {Fine-Grained Visual Classification of Aircraft},
   author        = {S. Maji and J. Kannala and E. Rahtu
                    and M. Blaschko and A. Vedaldi},
   year          = {2013},
   archivePrefix = {arXiv},
   eprint        = {1306.5151},
   primaryClass  = "cs-cv",
}
```
