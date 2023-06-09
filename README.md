<div align="center">
<h1> Benchmarking Robustness to Text-Guided Corruptions </h1>
<h3>

Mohammadreza Mofayezi and Yasamin Medghalchi </h3>

[![arXiv](https://img.shields.io/badge/paper-cvpr2023-gold)](https://openaccess.thecvf.com/content/CVPR2023W/GCV/html/Mofayezi_Benchmarking_Robustness_to_Text-Guided_Corruptions_CVPRW_2023_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-2304.02963-red)](https://arxiv.org/abs/2304.02963)

<image src="docs/Overview.png" width="600">
</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

*This study investigates the robustness of image classifiers to text-guided corruptions. We utilize diffusion models to edit images to different domains. Unlike other works that use synthetic or hand-picked data for benchmarking, we use diffusion models as they are generative models capable of learning to edit images while preserving their semantic content. Thus, the corruptions will be more realistic and the comparison will be more informative. Also, there is no need for manual labeling and we can create large-scale benchmarks with less effort. We define a prompt hierarchy based on the original ImageNet hierarchy to apply edits in different domains. As well as introducing a new benchmark we try to investigate the robustness of different vision models. The results of this study demonstrate that the performance of image classifiers decreases significantly in different language-based corruptions and edit domains. We also observe that convolutional models are more robust than transformer architectures. Additionally, we see that common data augmentation techniques can improve the performance on both the original data and the edited images. The findings of this research can help improve the design of image classifiers and contribute to the development of more robust machine learning systems.*

</br>

# Getting started

## Requirements
The code requires Python 3.8 or later. The file [requirements.txt](requirements.txt) contains the full list of required Python modules.
```bash
pip install -r requirements.txt
```

## Resources
The code was tested on a GeForce RTX 3090 24GB but should work on other cards with at least 12GB VRAM.

# Generating the Data
You can generate the text-guided benchmark using the command below:
```bash
python make_data.py --output_path data-100-10 --num_classes 100 --num_images 10 --sub_class all --seed 10
```

# Evaluating Models
You can run the evaluation code using the command bellow:
```bash
python main.py --data_path ./data-100-10/ --output_path data-100-10 --imagenet_path /imagenet/val/
```

# Acknowledgments

The overall code framework was adapted from [robustness](https://github.com/hendrycks/robustness).
The code for making image edits was borrowed from [prompt-to-prompt](https://github.com/google/prompt-to-prompt).

## Citation

```
@InProceedings{Mofayezi_2023_CVPR,
    author    = {Mofayezi, Mohammadreza and Medghalchi, Yasamin},
    title     = {Benchmarking Robustness to Text-Guided Corruptions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {779-786}
}
```
