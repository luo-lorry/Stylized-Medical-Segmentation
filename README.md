# Stylized-Medical-Segmentation
This repository contains the implementation of a novel medical image segmentation method that combines diffusion models and a Structure-Preserving Network for structure-aware one-shot image stylization. The proposed approach aims to mitigate domain shifts caused by variations in imaging devices, acquisition conditions, and patient-specific attributes, which often challenge the accuracy of medical image segmentation.
You can use link OSASIS(https://github.com/hansam95/OSASIS?tab=readme-ov-file#prepare-training) to transfer the style of the image. It is recommended to use the resize_and_stave_images function.
In the style transfer of polyp segmentation, we recommend training for 50 iterations, while in the style transfer of skin lesion segmentation, we recommend training for 30 iterations.
![figure2](https://github.com/user-attachments/assets/b64e6fcf-ca73-4efa-949a-f0a98f7b1943)

