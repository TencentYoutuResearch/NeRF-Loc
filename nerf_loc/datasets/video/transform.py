import torch
import cv2
import torch.nn as nn
from PIL import Image
import PIL
import numpy as np
import random
from .geometry import scale_K
import math
import torchvision

# __all__ = ['Resize', 'ResizeAndCrop', 'DownSample']

class Resize(object):
    def __init__(self, size):
        if type(size) is not tuple:
            self.h = size
            self.w = size
        else:
            self.h, self.w = size

    def __call__(self, img, depth, Tcw, K):
        h, w = img.shape[:2]
        # h, w = depth.shape

        scale_h = 1.0 * self.h / h
        scale_w = 1.0 * self.w / w
        K = scale_K(K, scale_w, scale_h)

        img = Image.fromarray(img)
        img = img.resize((self.w, self.h), resample=Image.ANTIALIAS)
        img = np.asarray(img)

        depth = Image.fromarray(depth)
        depth = depth.resize((self.w, self.h), resample=Image.NEAREST)
        depth = np.asarray(depth)

        return img, depth, Tcw, K

    def random_parameters(self):
        pass

class ResizeAndCrop(object):
    """
    Fit min(h,w) to target_size and crop to fit base_image_size
    """
    def __init__(self, target_size, base_image_size):
        assert target_size % base_image_size == 0
        self.target_size = target_size
        self.base_image_size = base_image_size

    def __call__(self, img, depth, Tcw, K, mask=None):
        h, w = img.shape[:2]
        # print(f'before: h {h} w {w} K {K}')
        if w > h:
            scale = 1.0 * self.target_size / h
        else:
            scale = 1.0 * self.target_size / w

        resize_h = int(round(scale * h))
        resize_w = int(round(scale * w))

        img = Image.fromarray(img)
        img = img.resize((resize_w, resize_h), resample=Image.ANTIALIAS)
        img = np.asarray(img)

        depth = Image.fromarray(depth)
        depth = depth.resize((resize_w, resize_h), resample=Image.NEAREST)
        depth = np.asarray(depth)

        if mask is not None:
            mask = Image.fromarray(mask)
            mask = mask.resize((resize_w, resize_h), resample=Image.NEAREST)
            mask = np.asarray(mask)

        K = scale_K(K, scale, scale)

        # crop
        padding_w = resize_w % self.base_image_size
        padding_h = resize_h % self.base_image_size
        if padding_w > 0:
            # crop along x axis
            img = img[:, padding_w//2:-(padding_w-padding_w//2)]
            depth = depth[:, padding_w//2:-(padding_w-padding_w//2)]
            if mask is not None:
                mask = mask[:, padding_w//2:-(padding_w-padding_w//2)]
        if padding_h > 0:
            # crop along y axis
            img = img[padding_h//2:-(padding_h-padding_h//2), :]
            depth = depth[padding_h//2:-(padding_h-padding_h//2), :]
            if mask is not None:
                mask = mask[padding_h//2:-(padding_h-padding_h//2), :]
        K[0, 2] = K[0, 2] - padding_w//2
        K[1, 2] = K[1, 2] - padding_h//2
        # print(f'after: h {img.shape[0]} w {img.shape[1]} K {K}')
        # if img.shape[0] == 0 or img.shape[1] == 0:
        #     from IPython import embed;embed()
        return img, depth, Tcw, K, mask

class DownSample(object):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, img, depth, Tcw, K):
        h, w = img.shape[:2]
        # h, w = depth.shape
        target_h = h // self.scale_factor
        target_w = w // self.scale_factor

        K = scale_K(K, target_w/w, target_h/h)

        img = Image.fromarray(img)
        img = img.resize((target_w, target_h), resample=Image.ANTIALIAS)
        img = np.asarray(img)

        depth = Image.fromarray(depth)
        depth = depth.resize((target_w, target_h), resample=Image.NEAREST)
        depth = np.asarray(depth)

        return img, depth, Tcw, K

    def random_parameters(self):
        pass

def zoom_image(img, scale_factor, interpolation=cv2.INTER_NEAREST):
    """
    scale image content without changing its size
    """
    h, w = img.shape[:2]
    target_h = int(h * scale_factor)
    target_w = int(w * scale_factor)
    content = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    if scale_factor > 1:
        # center crop
        padding_left = (target_w - w) // 2
        padding_right = target_w - w - padding_left
        padding_top = (target_h - h) // 2
        padding_bottom = target_h - h - padding_top
        img = content[padding_top:target_h-padding_bottom, padding_left:target_w-padding_right]
    elif scale_factor < 1:
        # padding
        padding_left = (w - target_w) // 2
        padding_right =  w - target_w - padding_left
        padding_top = (h - target_h) // 2
        padding_bottom = h - target_h - padding_top
        img = cv2.copyMakeBorder(
            content, padding_top, padding_bottom, padding_left, padding_right, 
            cv2.BORDER_CONSTANT, value=(0,0,0))
    assert img.shape[:2] == (h,w)
    sign = 1 if scale_factor < 1 else -1
    return img, sign * padding_left, sign * padding_top

class RandomZoom(object):
    def __init__(self, aug_scale_min, aug_scale_max):
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def __call__(self, img, depth, Tcw, K, mask=None):
        # h, w = img.shape[:2]
        # # h, w = depth.shape
        # target_h = int(h * self.scale_factor)
        # target_w = int(w * self.scale_factor)

        img, padding_left, padding_top = zoom_image(img, self.scale_factor, interpolation=cv2.INTER_LINEAR)
        depth, _, _ = zoom_image(depth, self.scale_factor, interpolation=cv2.INTER_NEAREST)
        if mask is not None:
            mask, _, _ = zoom_image(mask, self.scale_factor, interpolation=cv2.INTER_NEAREST)

        K = scale_K(K, self.scale_factor, self.scale_factor)
        K[0,2] += padding_left
        K[1,2] += padding_top

        # img = Image.fromarray(img)
        # img = img.resize((target_w, target_h), resample=Image.ANTIALIAS)
        # img = np.asarray(img)

        # depth = Image.fromarray(depth)
        # depth = depth.resize((target_w, target_h), resample=Image.NEAREST)
        # depth = np.asarray(depth)

        return img, depth, Tcw, K, mask

    def random_parameters(self):
        self.scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)


class RandomRotate(object):
    def __init__(self, aug_rotation):
        self.aug_rotation = aug_rotation # degree
        self.angle = 0

    def __call__(self, img, depth, Tcw, K, mask=None):
        img = Image.fromarray(img)
        img = img.rotate(self.angle, resample=Image.BICUBIC)
        img = np.asarray(img)

        depth = Image.fromarray(depth)
        depth = depth.rotate(self.angle, resample=Image.NEAREST)
        depth = np.asarray(depth)

        if mask is not None:
            mask = Image.fromarray(mask)
            mask = mask.rotate(self.angle, resample=Image.NEAREST)
            mask = np.asarray(mask)

        # rotate ground truth camera pose
        rad = -self.angle * math.pi / 180
        pose_rot = np.eye(4)
        pose_rot[0, 0] = math.cos(rad)
        pose_rot[0, 1] = -math.sin(rad)
        pose_rot[1, 0] = math.sin(rad)
        pose_rot[1, 1] = math.cos(rad)

        Tcw_hom = np.eye(4)
        Tcw_hom[:3] = Tcw

        Tcw = (pose_rot @ Tcw_hom)[:3]

        return img, depth, Tcw, K, mask

    def random_parameters(self):
        self.angle = random.uniform(-self.aug_rotation, self.aug_rotation)

class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        # self.op = torchvision.transforms.ColorJitter(
        #     brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img, depth, Tcw, K, mask=None):
        img = Image.fromarray(img)
        img = self.op(img)
        img = np.asarray(img)
        return img, depth, Tcw, K, mask

    def op(self, img):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.params

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = torchvision.transforms.functional.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = torchvision.transforms.functional.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = torchvision.transforms.functional.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = torchvision.transforms.functional.adjust_hue(img, hue_factor)

        return img

    def set_parameters(self, params):
        self.params = params

    def random_parameters(self):
        self.params = torchvision.transforms.ColorJitter.get_params(
            brightness=[1-self.brightness, 1+self.brightness], 
            contrast=[1-self.contrast, 1+self.contrast], 
            saturation=[1-self.saturation, 1+self.saturation], 
            hue=[-self.hue, self.hue])

class RandomCrop(object):
    def __init__(self, min_ratio=0.8, max_ratio=1):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img, depth, Tcw, K):
        if img is not None:
            h, w, _ = img.shape
            scale_ratio = (
                random.random() * (self.max_ratio - self.min_ratio) + self.min_ratio
            )
            left_up_ratio = random.random() * (1 - scale_ratio)
            x = int(w * left_up_ratio)
            y = int(h * left_up_ratio)

            new_h = int(h * scale_ratio)
            new_w = int(w * scale_ratio)

            img = img[y : y + new_h, x : x + new_w, :]
            depth = depth[y : y + new_h, x : x + new_w]

            K[0, 2] = K[0, 2] - x
            K[1, 2] = K[1, 2] - y

        return img, depth, Tcw, K

    def random_parameters(self):
        pass


class RandomCenterCrop(object):
    def __init__(self, min_ratio=0.8, max_ratio=1):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img, depth, Tcw, K):
        if img is not None:
            h, w, _ = img.shape

            g = math.gcd(h, w)
            unit_h = h / g
            unit_w = w / g

            x = int(int(w * (1 - self.scale_ratio) / 2 / unit_w) * unit_w)
            y = int(int(h * (1 - self.scale_ratio) / 2 / unit_h) * unit_h)

            img = img[y : h - y, x : w - x, :]
            depth = depth[y : h - y, x : w - x]

            K[0, 2] = (w - 2 * x) / 2
            K[1, 2] = (h - 2 * y) / 2

        return img, depth, Tcw, K

    def random_parameters(self):
        self.scale_ratio = (
            random.random() * (self.max_ratio - self.min_ratio) + self.min_ratio
        )


class CenterCrop(object):
    def __init__(self, scale_ratio=0.9):
        self.scale_ratio = scale_ratio

    def __call__(self, img, depth, Tcw, K):
        if img is not None:
            h, w, _ = img.shape

            x = int(w * (1 - self.scale_ratio) / 2)
            y = int(h * (1 - self.scale_ratio) / 2)

            new_h = int(h * self.scale_ratio)
            new_w = int(w * self.scale_ratio)
            # print(x,y,new_h,new_w)

            img = img[y : y + new_h, x : x + new_w, :]
            depth = depth[y : y + new_h, x : x + new_w]

            K[0, 2] = new_w / 2
            K[1, 2] = new_h / 2

        return img, depth, Tcw, K

    def random_parameters(self):
        pass


class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor(
                [
                    [0.4009, 0.7192, -0.5675],
                    [-0.8140, -0.0045, -0.5808],
                    [0.4203, -0.6948, -0.5836],
                ]
            )
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor, depth, Tcw, K):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros(*self.eig_val.size())) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)

        tensor = tensor + quatity.view(3, 1, 1)
        return tensor, depth, Tcw, K


class Normalize(object):
    def __init__(self, scale=1, mean=[0, 0, 0], std=[1, 1, 1]):
        # super(Resize).__init__()
        self.scale = scale
        self.mean = np.array(mean).reshape(1, 3)
        self.std = np.array(std).reshape(1, 3)

    def __call__(self, img, depth, Tcw, K):
        if img is not None:
            ori_shape = img.shape
            img = img.reshape(-1, 3)

            img = (img / self.scale - self.mean) / self.std
            img = img.reshape(ori_shape)
        return img, depth, Tcw, K

    def random_parameters(self):
        pass


class ToTensor(object):
    def __init__(self):
        # super(ToTensor).__init__()
        pass

    def __call__(self, img, depth, Tcw, K):

        return (
            torch.from_numpy(img).permute(2, 0, 1),
            torch.from_numpy(depth),
            torch.from_numpy(Tcw),
            torch.from_numpy(K),
        )

    def random_parameters(self):
        pass


class Compose(object):
    def __init__(self, transforms):
        # super(Compose).__init__()
        self.transforms = transforms

    def __call__(self, img, depth, Tcw, K, mask=None):
        for tsf in self.transforms:
            img, depth, Tcw, K, mask = tsf(img, depth, Tcw, K, mask)
        return img, depth, Tcw, K, mask

    def random_parameters(self):
        for tsf in self.transforms:
            tsf.random_parameters()
