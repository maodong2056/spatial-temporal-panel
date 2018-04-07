from pathlib import Path
from easydict import EasyDict as edict

# Dataset configuration
# ======================
DATASET = edict()
DATASET.root = Path.home() / 'Dataset/PANEL'
DATASET.video_root = DATASET.root / 'video'
DATASET.image_root = DATASET.root / 'image'
DATASET.label_root = DATASET.root / 'label'
DATASET.label_map = {
    # background index 0
    'other'  : 0,
    # standard panel
    'sp_sfb' : 1,  # ↘
    'sp_dn'  : 2,  # →→ AC
    'sp_cns' : 3,  # ↘
    'sp_qn'  : 4,  # →→ HC
    'sp_xn'  : 5,  # ↗
    'sp_gg'  : 6,  # →→ FE
    # basically standard
    # NOTE: here consider basic standard as absolute standard
    'bsp_sfb': 1,
    'bsp_dn' : 2,
    'bsp_cns': 3,
    'bsp_qn' : 4,
    'bsp_xn' : 5,
    'bsp_gg' : 6,
    # non-standard
    'nsp_sfb': 7,
    'nsp_dn' : 8,
    'nsp_cns': 9,
    'nsp_qn' : 10,
    'nsp_xn' : 11,
    'nsp_gg' : 12,
}
DATASET.train_split_num = {
    # train_nums = 1700 (400 + 900 + 400)
    # test_nums = 369 (70 + 184 + 115)

    'AC': 400,  # 400 / 470
    'FE': 800,  # 800 / 984
    'HC': 400,  # 400 / 515
}

# Training configuration
# ======================
TRAIN = edict()
TRAIN.batch_size = 1  # 3D volume
TRAIN.num_workers = 4
TRAIN.shuffle = True
