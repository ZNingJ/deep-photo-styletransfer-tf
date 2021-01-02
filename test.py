import numpy as np
from PIL import Image

VGG_MEAN = [103.939, 116.779, 123.68]

def rgb2bgr(rgb, vgg_mean=True):
    if vgg_mean:
        return rgb[:, :, ::-1] - VGG_MEAN
    else:
        return rgb[:, :, ::-1]

content_image_path='1c.png'

content_image = np.array(Image.open(content_image_path).convert("RGB"), dtype=np.float32)
content_width, content_height = content_image.shape[1], content_image.shape[0]

content_image = rgb2bgr(content_image)
content_image = content_image.reshape((1, content_height, content_width, 3)).astype(np.float32)

init_image = np.random.randn(1, content_height, content_width, 3).astype(np.float32)* 0.0001
print(init_image)
