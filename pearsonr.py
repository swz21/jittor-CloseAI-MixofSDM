import numpy as np
import blobfile as bf
import jittor as jt
from PIL import Image


def pearsonr(
        x: np.ndarray,
        y: np.ndarray,
        batch_first=True,
):
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, keepdims=True)
    y_std = y.std(axis=dim, keepdims=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr


def fast_color_histogram(image, color_stride):
    # input:image is a 3d array, shape[W][H][C]
    hist = np.arange((256 // color_stride) ** 3).reshape(256 // color_stride, 256 // color_stride, 256 // color_stride)
    H, W, C = image.shape
    image = np.array(image, dtype=np.int32) // color_stride
    index = image.reshape(-1, C).T

    hist = np.bincount(hist[index[0], index[1], index[2]], minlength=(256 // color_stride) ** 3)
    return hist

if __name__ == "__main__":
    with bf.BlobFile("output/jittor-138/3435300447_a996ea56bb_b.jpg", "rb") as f:
        ref_image = Image.open(f)
        ref_image.load()
    ref_image_2 = ref_image.convert("RGB")
    
    with bf.BlobFile("output/jittor-138/34360688026_d5b53b6447_b.jpg", "rb") as f:
        ref_image = Image.open(f)
        ref_image.load()
    ref_image = ref_image.convert("RGB")
    
    ref_image_jt = jt.Var(np.array(ref_image))
    
    p = pearsonr(fast_color_histogram(np.array(ref_image), 32).astype(np.float32), fast_color_histogram(np.array(ref_image_2), 32).astype(np.float32))
    print(p)