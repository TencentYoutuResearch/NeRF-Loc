import numpy as np
import cv2


def crop_from_center(img, new_h, new_w):
    """
    Crop the image with respect to the center
    :param img: image to be cropped
    :param new_h: cropped dimension on height
    :param new_w: cropped dimension on width
    :return: cropped image
    """

    h = img.shape[0]
    w = img.shape[1]
    x_c = w / 2
    y_c = h / 2

    crop_img = None
    if h >= new_h and w >= new_w:
        start_x = int(x_c - new_w / 2)
        start_y = int(y_c - new_h / 2)

        if len(img.shape) > 2:
            crop_img = img[
                start_y : start_y + int(new_h), start_x : start_x + int(new_w), :
            ]
        elif len(img.shape) == 2:
            crop_img = img[
                start_y : start_y + int(new_h), start_x : start_x + int(new_w)
            ]

    return crop_img


def fov(fx, fy, h, w):
    """
    Camera fov on x and y dimension
    :param fx: focal length on x axis
    :param fy: focal length on y axis
    :param h:  frame height
    :param w:  frame width
    :return: fov_x, fov_y
    """
    return (
        np.rad2deg(2 * np.arctan(w / (2 * fx))),
        np.rad2deg(2 * np.arctan(h / (2 * fy))),
    )


def crop_by_intrinsic(img, cur_k, new_k, interp_method="bilinear"):
    """
    Crop the image with new intrinsic parameters
    :param img: image to be cropped
    :param cur_k: current intrinsic parameters, 3x3 matrix
    :param new_k: crop target intrinsic parameters, 3x3 matrix
    :return: cropped image
    """
    cur_fov_x, cur_fov_y = fov(
        cur_k[0, 0], cur_k[1, 1], 2 * cur_k[1, 2], 2 * cur_k[0, 2]
    )
    new_fov_x, new_fov_y = fov(
        new_k[0, 0], new_k[1, 1], 2 * new_k[1, 2], 2 * new_k[0, 2]
    )
    crop_img = None
    if cur_fov_x >= new_fov_x and cur_fov_y >= new_fov_y:
        # Only allow to crop to a smaller fov image
        # 1. Resize image
        focal_ratio = new_k[0, 0] / cur_k[0, 0]
        if interp_method == "nearest":
            crop_img = cv2.resize(
                img,
                (int(focal_ratio * img.shape[1]), int(focal_ratio * img.shape[0])),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            crop_img = cv2.resize(
                img, (int(focal_ratio * img.shape[1]), int(focal_ratio * img.shape[0]))
            )

        # Crop the image with new w/h ratio with respect to the center
        crop_img = crop_from_center(crop_img, 2 * new_k[1, 2], 2 * new_k[0, 2])
    else:
        raise Exception("The new camera FOV is larger then the current.")

    return crop_img
