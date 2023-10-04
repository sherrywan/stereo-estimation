import cv2
import numpy as np
import os, sys
import io
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
from functools import partial
from rembg import remove

sys.path.append(os.path.join("/data0/wxy/3d_pose/stereo-estimation/"))
from lib.dataset.human36m import Human36MMultiViewDataset

def load_img(img_file):

    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if not img_file.endswith("png"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img

def get_image_mask(img_ori):

    mask_to_origin_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0, ), (1.0, ))
    ])
    with torch.no_grad():
        buf = io.BytesIO()
        Image.fromarray(img_ori).save(buf, format='png')
        img_pil = Image.open(
            io.BytesIO(remove(buf.getvalue()))).convert("RGBA")
    img_mask = torch.tensor(1.0) - (mask_to_origin_tensor(img_pil.split()[-1]) <
                                    torch.tensor(0.5)).float()

    return img_mask

def apply_mask(image, mask):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 0,
                                  255,
                                  image[:, :, c])
    return image


def gen_mask(idx, h36m_root=None):
    sample = dataset[idx]
        
    if sample.get('images') is None:
        return
        
    shot = dataset.labels['table'][idx]
    subject_idx = shot['subject_idx']
    action_idx  = shot['action_idx']
    frame_idx   = shot['frame_idx']

    subject = dataset.labels['subject_names'][subject_idx]
    action = dataset.labels['action_names'][action_idx]
    
    available_cameras = list(range(len(dataset.labels['camera_names'])))
    
    for camera_idx, bbox in enumerate(shot['bbox_by_camera_tlbr']):
        if bbox[2] == bbox[0]: # bbox is empty, which means that this camera is missing
            available_cameras.remove(camera_idx)
    
    for camera_idx, image, silouette, bbox in zip(available_cameras, sample['images'], sample["silhouettes"], sample['detections']):
        h, w = silouette.shape
        if (h==w) and (h == (bbox[2]-bbox[0])):
            continue
        img = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        mask = get_image_mask(img)
        camera_name = dataset.labels['camera_names'][camera_idx]
        image_path = os.path.join(
                    h36m_root, subject, action,
                    'imageSequence-silhouette' + '-croped',
                    camera_name, 'img_%06d.jpg' % (frame_idx + 1))
        os.makedirs(os.path.join(
                    h36m_root, subject, action,
                    'imageSequence-silhouette' + '-croped'), exist_ok=True)
        os.makedirs(os.path.join(
                    h36m_root, subject, action,
                    'imageSequence-silhouette' + '-croped', camera_name), exist_ok=True)

        cv2.imwrite(image_path, np.squeeze(mask.numpy(),0)*255)
        # print("save mask to {}".format(image_path))


h36m_root = os.path.join("/data1/share/dataset/human36m_multi-view/", "processed")
labels_stereo_npy_path = "/data1/share/dataset/human36m_multi-view/extra/human36m-stereo-labels-GTbboxes.npy"

dataset = Human36MMultiViewDataset(
    h36m_root,
    labels_stereo_npy_path,
    train=True,                       # include all possible data
    test=True,
    image_shape=None,                 # don't resize
    retain_every_n_frames_in_test=1,  # yes actually ALL possible data
    with_damaged_actions=True,        # I said ALL DATA
    kind="mpii",
    norm_image=False,                 # don't do unnecessary image processing
    undistort_images=True,                 
    crop=True)                       # don't crop
print("Dataset length:", len(dataset))

number_of_processes = 20
print(f"Crop and mask images using {number_of_processes} parallel processes")
cv2.setNumThreads(1)
import multiprocessing
pool = multiprocessing.Pool(number_of_processes)
partial_work = partial(gen_mask, h36m_root=h36m_root)
for _ in tqdm(pool.imap_unordered(
    partial_work, range(len(dataset)), chunksize=1), total=len(dataset)):
    pass

pool.close()
pool.join()