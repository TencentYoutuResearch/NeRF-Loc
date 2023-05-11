import lmdb
import sys
import numpy as np
import cv2
import os.path as osp
import os
import shutil
from skimage.io import imread
import h5py

class LMDBModel:

    # Path to the LMDB
    lmdb_path = None

    # LMDB Environment handle
    __lmdb_env__ = None

    # LMDB context handle
    __lmdb_txn__ = None

    # LMDB Cursor for navigating data
    __lmdb_cursor__ = None

    """ Constructor and De-constructor
    """

    def __init__(self, lmdb_path, workers=3):
        self.lmdb_path = lmdb_path
        self.__start_session__(workers=workers)

    def __del__(self):
        self.close_session()

    """ Session Function
    """

    def __start_session__(self, workers):

        # Open LMDB file
        self.__lmdb_env__ = lmdb.open(
            self.lmdb_path, max_readers=workers, readonly=True
        )

        # Crete context
        self.__lmdb_txn__ = self.__lmdb_env__.begin(write=False)

        # Get the cursor of current lmdb
        self.__lmdb_cursor__ = self.__lmdb_txn__.cursor()

    def close_session(self):
        if self.__lmdb_env__ is not None:
            self.__lmdb_env__.close()
            self.__lmdb_env__ = None

    """ Read Routines
    """

    def read_by_key(self, key):

        """
        Read value in lmdb by providing the key
        :param key: the string that corresponding to the value
        :return: array data
        """
        value = self.__lmdb_cursor__.get(key.encode())
        return value

    def read_ndarray_by_key(self, key, dtype=np.float32):
        value = self.__lmdb_cursor__.get(key.encode())
        return np.fromstring(value, dtype=dtype)

    def len_entries(self):
        length = self.__lmdb_txn__.stat()["entries"]
        return length

    """ Static Utilities
    """

    @staticmethod
    def convert_to_img(data):

        """
        Transpose the data from the Caffe's format to the normal format
        :param data: ndarray object with dimension of (3, h, w)
        :return: transposed ndarray with dimension of (h, w, 3)
        """
        return data.transpose((1, 2, 0))

    def get_keys(self):
        keys = []
        for key, value in self.__lmdb_cursor__:
            keys.append(key)
        return keys


class LMDBWriter:

    """ Write the dataset to LMDB database
    """

    """ Variables
    """
    __key_counts__ = 0

    # LMDB environment handle
    __lmdb_env__ = None

    # LMDB context handle
    __lmdb_txn__ = None

    # LMDB Path
    lmdb_path = None

    """ Functions
    """

    def __init__(self, lmdb_path, auto_start=True):
        self.lmdb_path = lmdb_path
        self.__del_and_create__(lmdb_path)
        if auto_start is True:
            self.__start_session__()

    def __del__(self):
        self.close_session()

    def __del_and_create__(self, lmdb_path):
        """
        Delete the exist lmdb database and create new lmdb database.
        """
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        os.mkdir(lmdb_path)

    def __start_session__(self):
        self.__lmdb_env__ = lmdb.Environment(self.lmdb_path, map_size=1099511627776)
        self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True)

    def close_session(self):
        if self.__lmdb_env__ is not None:
            self.__lmdb_txn__.commit()
            self.__lmdb_env__.close()
            self.__lmdb_env__ = None
            self.__lmdb_txn__ = None

    def write_str(self, key, str):
        """
        Write the str data to the LMDB
        :param key: key in string type
        :param array: array data
        """
        # Put to lmdb
        self.__key_counts__ += 1
        self.__lmdb_txn__.put(key.encode(), str)
        if self.__key_counts__ % 10000 == 0:
            self.__lmdb_txn__.commit()
            self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True, buffers=True)

    def write_array(self, key, array):
        """
        Write the array data to the LMDB
        :param key: key in string type
        :param array: array data
        """
        # Put to lmdb
        self.__key_counts__ += 1
        self.__lmdb_txn__.put(key.encode(), array.tostring())
        if self.__key_counts__ % 10000 == 0:
            self.__lmdb_txn__.commit()
            self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True, buffers=True)


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def load_extrinsic(meta_info):
    if len(meta_info["extrinsic_Tcw"]) == 16:
        Tcw = np.array(meta_info["extrinsic_Tcw"]).reshape(4, 4)
        Tcw = Tcw[:3, :]
    else:
        Tcw = np.array(meta_info["extrinsic_Tcw"]).reshape(3, 4)
    return Tcw


def load_rgb_intrinsic(meta_info):
    # if dataset=='default':
    K_param = meta_info["camera_intrinsic"]
    K = np.zeros((3, 3))
    K[0, 0] = K_param[0]
    K[1, 1] = K_param[1]
    K[2, 2] = 1
    K[0, 2] = K_param[2]
    K[1, 2] = K_param[3]
    return K

# def load_depth_intrinsic(meta_info):
#     # if dataset=='default':
#     K_param = meta_info["depth_intrinsic"]
#     K = np.zeros((3, 3))
#     K[0, 0] = K_param[0]
#     K[1, 1] = K_param[1]
#     K[2, 2] = 1
#     K[0, 2] = K_param[2]
#     K[1, 2] = K_param[3]
#     return K

def load_depth_map(file_path):
    if file_path.endswith('.geometric.bin'):
        depth = read_array(file_path) * 1000
    elif file_path.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            depth = (f['depth'][()] * 1000).astype(np.uint16)
    else:
        # from depth camera
        depth = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        depth[depth==65535] = 0
    return depth


def load_one_img(base_dir, meta_info):
    Tcw = load_extrinsic(meta_info)
    K_rgb = load_rgb_intrinsic(meta_info)
    # H = int(K[1, 2] * 2)
    # W = int(K[0, 2] * 2)
    # K_depth = load_depth_intrinsic(meta_info)

    file_name = meta_info["file_name"]
    depth_file_name = meta_info["depth_file_name"]

    img = None

    img_path = osp.join(base_dir, file_name)
    depth_file_name = osp.join(base_dir, depth_file_name)

    img = imread(img_path)
    H,W = img.shape[:2]
    # TODO: test frame in camdbridge dataset have no depth
    if os.path.exists(depth_file_name):
        depth = load_depth_map(depth_file_name)
    else:
        depth = np.zeros([H,W])

    # resize depth to rgb size
    H_d, W_d = depth.shape[:2]
    sh, sw = float(H) / H_d, float(W) / W_d
    # K_depth[0] *= sw
    # K_depth[1] *= sh
    try:
        depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)
    except:
        from IPython import embed;embed()

    depth=depth.astype(np.float32)/1000
    depth[depth < 1e-5] = 0
    return img, depth, Tcw, K_rgb
