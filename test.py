"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import jittor as jt
import numpy as np
import json
import blobfile as bf
from PIL import Image

from jittor_guided_diffusion.image_datasets import load_data

from jittor_guided_diffusion import logger  # dist_util
from jittor_guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from pearsonr import pearsonr, fast_color_histogram

jt.flags.use_cuda = 1 # jt.flags.use_cuda 表示是否使用 gpu 训练。

def main():
    args = create_argparser().parse_args()

    n_moe_samples = 40

    if os.path.exists("./label_to_img.json"):
        with open("./label_to_img.json", 'r', encoding="utf-8") as file:
            label_to_img = json.load(file)
    else:
        print("[Error] label_to_img.json not found.")
        exit()

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    params = jt.load(args.model_path)
    model.load_state_dict(params)

    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.input_path,
        batch_size=args.batch_size,
        image_width=args.image_width,
        image_height=args.image_height,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        is_train=False,
        label_dir=args.img_path
    )

    model.eval()

    logger.log("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        image = ((batch + 1.0) / 2.0)
        model_kwargs = preprocess_input(cond, num_classes=args.num_classes)

        # set hyperparameter
        model_kwargs['s'] = args.s

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        samples = []
        for idx in range(args.batch_size):
            samples.append([])
        for _ in range(n_moe_samples):
            sample = sample_fn(
                model,
                (args.batch_size, 3, image.shape[2], image.shape[3]),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True
            )
            sample = (sample + 1) / 2.0
            for idx in range(args.batch_size):
                samples[idx].append(sample[idx])

        for idx in range(args.batch_size):
            label_name = cond['path'][idx].split('/')[-1].split('.')[0] + '.png'
            # assert label_name in label_to_img  # wrong ref path?
            ref_name = label_to_img[label_name]

            with bf.BlobFile(os.path.join(args.input_path, "imgs", ref_name), "rb") as f:
                ref_image = Image.open(f)
                ref_image.load()
            ref_image = ref_image.convert("RGB")
            ref_histogram = fast_color_histogram(np.array(ref_image), 32).astype(np.float32)

            diffusion_pearsonrs = [pearsonr(fast_color_histogram(diffusion_image.numpy(), 32).astype(np.float32), ref_histogram) for diffusion_image in samples[idx]]

            max_diffusion_pearsonr = max(diffusion_pearsonrs)
            max_index = diffusion_pearsonrs.index(max_diffusion_pearsonr)
            sample[idx] = samples[idx][max_index]

        gathered_samples = [sample]
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        for j in range(sample.shape[0]):
            jt.misc.save_image(sample[j], os.path.join(args.output_path, cond['path'][j].split('/')[-1].split('.')[0] + '.jpg'))

        logger.log(f"created {len(all_samples) * args.batch_size} samples")

        if len(all_samples) * args.batch_size > args.num_samples:
            break

    logger.log("sampling complete")


def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    # input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    input_label = jt.init.zero((bs, num_classes, h, w))
    input_semantics = input_label.scatter_(1, label_map, jt.Var([1.0]))

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = jt.concat((input_semantics, instance_edge_map), dim=1)

    return {'y': input_semantics}


def get_edges(t):
    edge = jt.init.zero(t.size()).int8()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def create_argparser():
    defaults = dict(
        input_path="",
        img_path="",
        dataset_mode="jittor",
        clip_denoised=True,
        num_samples=1000,
        batch_size=2,
        use_ddim=False,
        model_path="./ema_0.9999_1380000.jt",
        output_path="",
        is_train=False,
        s=1.5,
        ddim_steps=0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
