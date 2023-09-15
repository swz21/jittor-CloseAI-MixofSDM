# Semantic Image Synthesis via Diffusion Models (SDM)

## Training and Test

We use python3.9.

- Train the SDM model with Jittor:
  
  ```bash
  python image_train.py --input_path /root/data/train_resized
  ```

- Test the SDM model with Jittor:
  
  ```bash
  python image_sample.py --input_path /root/data/train_resized --img_path /root/data/val_B_labels_resized --output_path ./output
  ```

### Acknowledge

Our code is developed based on [WeilunWang/semantic-diffusion-model: Official Implementation of Semantic Image Synthesis via Diffusion Models (github.com)](https://github.com/WeilunWang/semantic-diffusion-model).

[Paper](https://arxiv.org/abs/2207.00050)

[Weilun Wang](https://scholar.google.com/citations?hl=zh-CN&user=YfV4aCQAAAAJ), [Jianmin Bao](https://scholar.google.com/citations?hl=zh-CN&user=hjwvkYUAAAAJ), [Wengang Zhou](https://scholar.google.com/citations?hl=zh-CN&user=8s1JF8YAAAAJ), [Dongdong Chen](https://scholar.google.com/citations?hl=zh-CN&user=sYKpKqEAAAAJ), [Dong Chen](https://scholar.google.com/citations?hl=zh-CN&user=_fKSYOwAAAAJ), [Lu Yuan](https://scholar.google.com/citations?hl=zh-CN&user=k9TsUVsAAAAJ), [Houqiang Li](https://scholar.google.com/citations?hl=zh-CN&user=7sFMIKoAAAAJ)