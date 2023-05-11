import numpy as np
import math
import sys
import cv2

# sys.path.append("libs/utils/lm_pnp/build")
# import lm_pnp


def x_2d_coords(h, w):
    x_2d = np.zeros((h, w, 2), dtype=np.float32)
    for y in range(0, h):
        x_2d[y, :, 1] = y
    for x in range(0, w):
        x_2d[:, x, 0] = x
    return x_2d


def vector_norm(data, axis=None, out=None):
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def quaternion_about_axis(angle, axis):
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen = vector_norm(q)
    if qlen > np.finfo(float).eps * 4.0:
        q *= math.sin(angle / 2.0) / qlen
    q[0] = math.cos(angle / 2.0)
    return q


def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def gen_rot_mat(min_v=None, max_v=None):
    if min_v is None:
        rand_angle = np.random.random_sample() * 2.0 * np.pi
    else:
        rand_angle = ((max_v - min_v) * np.random.random_sample() + min_v) * 2.0 * np.pi

    rand_R = quaternion_matrix(quaternion_about_axis(rand_angle, (0.0, 1.0, 0.0)))[
        :3, :3
    ].astype(np.float32)
    return rand_R


def scale_K(K, fx, fy):
    # K = K * rescale_factor
    K[0,:] *= fx
    K[1,:] *= fy
    K[2, 2] = 1.0
    return K


def rel_rot_quaternion_deg(q1, q2):
    """
    Compute relative error (deg) of two quaternion
    :param q1: quaternion 1, (w, x, y, z), dim: (4)
    :param q2: quaternion 2, (w, x, y, z), dim: (4)
    :return: relative angle in deg
    """
    return 2 * 180 * np.arccos(np.clip(np.dot(q1, q2), -1.0, 1.0)) / np.pi


def rel_rot_angle(T1, T2):
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    q1 = quaternion_from_matrix(R1)
    q2 = quaternion_from_matrix(R2)
    return rel_rot_quaternion_deg(q1, q2)


def rel_distance(T1, T2):
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    d = np.dot(R1.T, t1) - np.dot(R2.T, t2)
    return np.linalg.norm(d)


def pi_inv(K, x, d):
    fx, fy, cx, cy = K[0:1, 0:1], K[1:2, 1:2], K[0:1, 2:3], K[1:2, 2:3]
    X_x = d * (x[:, :, 0] - cx) / fx
    X_y = d * (x[:, :, 1] - cy) / fy
    X_z = d
    X = np.stack([X_x, X_y, X_z], axis=0).transpose([1, 2, 0])
    return X


def inv_pose(R, t):
    Rwc = R.T
    tw = -Rwc.dot(t)
    return Rwc, tw


def transpose(R, t, X):
    X = X.reshape(-1, 3)
    X_after_R = R.dot(X.T).T
    trans_X = X_after_R + t
    return trans_X


def back_projection(depth, Tcw, K):
    h, w = depth.shape
    x_2d = x_2d_coords(h, w)

    X_3d = pi_inv(K, x_2d, depth)
    Rwc, twc = inv_pose(R=Tcw[:3, :3], t=Tcw[:3, 3])
    X_world = transpose(Rwc, twc, X_3d)

    X_world = X_world.reshape((h, w, 3))
    return X_world


def projection(coords, P, get_depth=False):
    H, W, _ = coords.shape

    coords = coords.reshape(-1, 3)
    ones = np.ones((H * W, 1))

    coords_homo = np.concatenate([coords, ones], axis=1).T

    coords_2d = P @ coords_homo

    mask = (coords_2d[-1, :] > 0.1).reshape(H, W)
    coords_2d = coords_2d[:2, :] / coords_2d[-1:, :]

    if get_depth:
        return coords_2d[-1:, :].reshape(H, W)
    coords_2d = coords_2d.reshape(2, H, W)

    mask = (
        mask
        * (coords_2d[0, :, :] >= 0)
        * (coords_2d[0, :, :] < W)
        * (coords_2d[1, :, :] >= 0)
        * (coords_2d[1, :, :] < H)
    )
    return coords_2d, mask


def compute_pose_lm_pnp(
    query_X_w, pnp_x_2d, query_K, repro_thres, hypo=128, refine_steps=100, verbose=False
):
    # recover original scene coordinates
    # query_X_w=query_X_w.reshape(-1,3)
    # query_X_3d_w = query_X_w.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()

    lm_pnp_pose_vec, inlier_map = lm_pnp.compute_lm_pnp(
        pnp_x_2d.reshape(1, -1, 2),
        query_X_w.reshape(1, -1, 3),
        query_K,
        repro_thres,
        hypo,
        refine_steps,
        verbose,
    )

    R_res, _ = cv2.Rodrigues(lm_pnp_pose_vec[:3])
    lm_pnp_pose = np.zeros((3, 4))
    lm_pnp_pose[:3, :3] = R_res
    lm_pnp_pose[:3, 3] = lm_pnp_pose_vec[3:].ravel()

    return lm_pnp_pose, inlier_map
