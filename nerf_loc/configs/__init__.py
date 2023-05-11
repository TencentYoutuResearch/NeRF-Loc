from yacs.config import CfgNode as CN


_C = CN()
_C.expname = ''
_C.basedir = ''
_C.datadir = ''
_C.version = 'default'
_C.ckpt = ''
_C.dataset_type = 'video_cambridge'
_C.scenes = []

_C.max_epochs = 50
_C.lrate = 5e-4
_C.lrate_decay_steps = 50000
_C.lrate_decay_factor = 0.5

_C.train_nerf = True
_C.train_pose = True

_C.backbone2d = 'cotr'
_C.backbone2d_fpn_dim = 192
_C.backbone2d_use_fpn = True
_C.backbone2d_coarse_layer_name = 'layer2'
_C.backbone2d_fine_layer_name = 'layer1'

# support image
_C.support_image_selection = 'retrieval'
_C.n_views_train = 5
_C.n_views_test = 10
_C.image_core_set_size = 16
# image retrieval
_C.image_retrieval_method = 'netvlad' # used in offline preprocessing
_C.image_retrieval_method_train = 'netvlad'
_C.image_retrieval_method_test = 'netvlad'
_C.image_retrieval_interval_train = 1
_C.image_retrieval_interval_test = 1
# coreset
_C.coreset_sampler = 'FPS'

_C.model_3d_hidden_dim = 128
_C.use_scene_coord_memorization = False

_C.encode_appearance = True
_C.appearance_emb_dim = 128

_C.simple_3d_model = False

# position embedding
_C.multires = 10 # log2 of max freq for positional encoding (3D location)
_C.multires_views = 4 # log2 of max freq for positional encoding (2D direction)
_C.i_embed = 0 # set 0 for default positional encoding, -1 for none

_C.render = CN()
_C.render.N_samples = 64
_C.render.N_importance = 0
_C.render.N_rand = 1024
_C.render.chunk = 2048
_C.render.lindisp = False
_C.render.white_bkgd = False
_C.render.use_render_uncertainty = True
_C.render.render_feature = True

_C.use_depth_supervision = False
_C.coarse_loss_weight = 10000.
_C.fine_loss_weight = 10.
_C.render_loss_weight = 1.0
_C.ref_depth_loss_weight = 0.1

_C.keypoints_3d_source = 'depth' # sfm - from sparse sfm points, depth - from dense backprojected points
_C.matcher_hidden_dim = 192
_C.matching = CN()
_C.matching.keypoints_3d_sampling = 'random'
_C.matching.keypoints_3d_sampling_max_keep = 100000
_C.matching.coarse_matching_depth_thresh = 2.
_C.matching.coarse_num_3d_keypoints = 1024
_C.matching.fine_num_3d_keypoints = 1024
_C.fine_matching_loss_type = 'l2_with_std'

_C.ransac_thresh = 8
_C.rotation_eval_thresh = 5
_C.translation_eval_thresh = 0.05

# test time
_C.cascade_matching = False
_C.optimize_pose = False
_C.test_time_color_jitter = False
_C.test_time_style_change = False
_C.test_render_interval = 50 # interval of rendering test image
_C.vis_3d_box = False # save onepose box visualization
_C.vis_rendering = False # save rendered image for visualization
_C.vis_trajectory = False # save camera trajectory for visualization

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    return _C.clone()

def override_cfg_with_args(cfg, args):
    for name in vars(args):
        if name in cfg:
            setattr(cfg, name, getattr(args, name))
    return cfg
