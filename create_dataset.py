import os
import shutil
import imageio
import scipy.io as sio
from tqdm import tqdm
from pathlib import Path

from joblib import Parallel, delayed
from skimage import img_as_ubyte, io, color
from configuration import DATASET


def video_to_images_worker(video_path, out_root):
    vp = video_path
    image_root = out_root / vp.stem
    image_root.mkdir(exist_ok=True)

    reader = imageio.get_reader(str(vp))
    for idx, im in enumerate(reader, start=1):
        cropped = im[100:900, 250:1050]
        gray = color.rgb2gray(cropped)
        out_name = image_root / '{}_{:03d}.jpg'.format(vp.stem, idx)
        io.imsave(out_name, img_as_ubyte(gray))


def video_to_images_master(video_root, out_root):
    video_paths = sorted(video_root.glob('*/*'))
    Parallel(n_jobs=48, verbose=50)(delayed(video_to_images_worker)(vp, out_root) for vp in video_paths)


def parse_labels_per_video(label_root, video_name):
    """
    Args:
        video_name: e.g., KLFE0512
    Return:
        <list> of [frame_name, label] pair,
        e.g., [['KLFE0512_1.jpg', 'other'],
               ['KLFE0512_2.jpg', 'other'],
               ['KLFE0512_3.jpg', 'nsp_gg'],
               ['KLFE0512_4.jpg', 'sp_gg'],
               ['KLFE0512_5.jpg', 'bsp_gg']...]
    """
    label_path = label_root / video_name[2:4] / 'mat' / '{}.mat'.format(video_name)
    if not label_path.exists():
        return None
    mat_contents = sio.loadmat(str(label_path))['label'].squeeze()
    str_contents = [['{}_{:03d}.jpg'.format(video_name, i), str(mc[0])]
                    for i, mc in enumerate(mat_contents, start=1)]
    return str_contents


def trans_name(name):
    """Transform name
    E.g.,
    `KLAC0001_1.jpg` --> KLAC0001_001.jpg
    """
    name = str(name)
    index = int(name.split('_')[-1].split('.')[0])
    return '{}_{:03d}.jpg'.format(name.split('_')[0], index)


def create_list(out_root, image_root, label_root, video_names):
    label_not_found = 0

    for video_name in tqdm(video_names):
        label_pairs = parse_labels_per_video(label_root, video_name)
        if label_pairs is None:
            print('[Label] {} Not Found'.format(video_name))
            label_not_found += 1
            continue
        out_name = out_root / video_name
        with open(out_name.with_suffix('.txt'), 'w+') as f:
            for frame_name, label in label_pairs:
                frame_path = image_root / video_name / frame_name
                if not frame_path.exists():
                    print('[Frame]: {} Not Found'.format(frame_path))
                    break
                label_digit = DATASET.label_map[label]
                f.write('{} {}\n'.format(frame_path, label_digit))

    # print('Total {} labels not found'.format(label_not_found))


def create_train_test_split(list_root):
    train_root = list_root / 'train'
    test_root = list_root / 'test'
    [r.mkdir(exist_ok=True) for r in [train_root, test_root]]

    for category, train_num in DATASET.train_split_num.items():
        category_list = sorted(list_root.glob('KL{}*.txt'.format(category)))
        [shutil.copyfile(r, train_root / r.name) for r in category_list[:train_num]]
        [shutil.copyfile(r, test_root / r.name) for r in category_list[train_num:]]
        [os.remove(r) for r in category_list]


def main():
    # 1. convert videos to images
    videos = sorted([i.stem for i in list(DATASET.video_root.glob('*/*'))])
    images = sorted([i.stem for i in list(DATASET.image_root.iterdir())])
    if not videos == images:  # check images exist
        video_to_images_master(DATASET.video_root, DATASET.image_root)

    # 2. extract labels and make list
    out_root = Path('list')
    out_root.mkdir(exist_ok=True)
    create_list(out_root, DATASET.image_root, DATASET.label_root, videos)

    # 3. split train/test dataset
    create_train_test_split(out_root)


if __name__ == '__main__':
    main()
