# Stylized-Medical-Segmentation
This repository contains the implementation of a novel medical image segmentation method that combines diffusion models and a Structure-Preserving Network for structure-aware one-shot image stylization. The proposed approach aims to mitigate domain shifts caused by variations in imaging devices, acquisition conditions, and patient-specific attributes, which often challenge the accuracy of medical image segmentation.
You can use link OSASIS(https://github.com/hansam95/OSASIS?tab=readme-ov-file#prepare-training) to transfer the style of the image. It is recommended to use the resize_and_stave_images function.
In the style transfer of polyp segmentation, we recommend training for 50 iterations, while in the style transfer of skin lesion segmentation, we recommend training for 30 iterations.
![figure2](https://github.com/user-attachments/assets/b64e6fcf-ca73-4efa-949a-f0a98f7b1943)
We propose a novel approach that integrates diffusion-based style transfer with image segmentation to enhance the robustness of medical image segmentation under domain shifts.
![figure3](https://github.com/user-attachments/assets/6ad0e8b6-2d36-4dfd-9bd4-9cd6a079a89c)
If our ideas are helpful to you, please cite our article:

@article{Jie2024Structure,

title={Structure-Aware Stylized Image Synthesis for Robust Medical Image Segmentation},

author={Bao, Jie and Zhou, Zhixin Zhou and Li, Wen Jung and Luo, Rui},

journal={arXiv preprint arXiv:2412.04296},

year={2024}

}
