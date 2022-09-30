import numpy as np
import scipy.ndimage
import skimage.transform
import cv2

import torch

import matplotlib

matplotlib.use('Agg')
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.pyplot import MultipleLocator

from lib.utils.img import image_batch_to_numpy, to_numpy, denormalize_image, resize_image
from lib.utils.multiview import project_3d_points_to_image_plane_without_distortion

CONNECTIVITY_DICT = {
    'cmu': [(0, 2), (0, 9), (1, 0), (1, 17), (2, 12), (3, 0), (4, 3), (5, 4),
            (6, 2), (7, 6), (8, 7), (9, 10), (10, 11), (12, 13), (13, 14),
            (15, 1), (16, 15), (17, 18)],
    'coco': [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
             (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (5, 11), (6, 12),
             (11, 12)],
    "mpii": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8),
             (8, 9), (8, 12), (8, 13), (10, 11), (11, 12), (13, 14), (14, 15)],
    "human36m": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6),
                 (6, 7), (7, 8), (8, 16), (9, 16), (8, 12), (11, 12), (10, 11),
                 (8, 13), (13, 14), (14, 15)],
    "kth": [(0, 1), (1, 2), (5, 4), (4, 3), (6, 7), (7, 8), (11, 10), (10, 9),
            (2, 3), (3, 9), (2, 8), (9, 12), (8, 12), (12, 13)],
}

COLOR_DICT = {
    'coco': [
        (102, 0, 153),
        (153, 0, 102),
        (51, 0, 153),
        (153, 0, 153),  # head
        (51, 153, 0),
        (0, 153, 0),  # left arm
        (153, 102, 0),
        (153, 153, 0),  # right arm
        (0, 51, 153),
        (0, 0, 153),  # left leg
        (0, 153, 102),
        (0, 153, 153),  # right leg
        (153, 0, 0),
        (153, 0, 0),
        (153, 0, 0),
        (153, 0, 0)  # body
    ],
    'human36m': [
        (0, 153, 102),
        (0, 153, 153),
        (0, 153, 153),  # right leg
        (0, 51, 153),
        (0, 0, 153),
        (0, 0, 153),  # left leg
        (153, 0, 0),
        (153, 0, 0),  # body
        (153, 0, 102),
        (153, 0, 102),  # head
        (153, 153, 0),
        (153, 153, 0),
        (153, 102, 0),  # right arm
        (0, 153, 0),
        (0, 153, 0),
        (51, 153, 0)  # left arm
    ],
    'kth': [
        (0, 153, 102),
        (0, 153, 153),  # right leg
        (0, 51, 153),
        (0, 0, 153),  # left leg
        (153, 102, 0),
        (153, 153, 0),  # right arm
        (51, 153, 0),
        (0, 153, 0),  # left arm
        (153, 0, 0),
        (153, 0, 0),
        (153, 0, 0),
        (153, 0, 0),
        (153, 0, 0),  # body
        (102, 0, 153)  # head
    ]
}

JOINT_NAMES_DICT = {
    'coco': {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
}


def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)

    return fig_image


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 3D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    # canvas.tostring_rgb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGB", (w, h), buf.tostring())
    image = np.asarray(image)

    return image


def visualize_batch_alg(images_batch,
                        heatmaps_batch,
                        keypoints_2d_batch,
                        proj_matricies_batch,
                        keypoints_3d_batch_gt,
                        keypoints_3d_batch_pred,
                        kind="cmu",
                        cuboids_batch=None,
                        confidences_batch=None,
                        batch_index=0,
                        size=5,
                        max_n_cols=10,
                        pred_kind=None):
    if pred_kind is None:
        pred_kind = kind

    n_views, n_joints = heatmaps_batch.shape[1], heatmaps_batch.shape[2]

    n_rows = 1
    # n_rows = n_rows + 1 if keypoints_2d_batch is not None else n_rows
    n_rows = n_rows + 1 if cuboids_batch is not None else n_rows
    n_rows = n_rows + 1 if confidences_batch is not None else n_rows

    n_cols = min(n_views, max_n_cols)
    fig, axes = plt.subplots(ncols=n_cols,
                             nrows=n_rows,
                             figsize=(n_cols * size, n_rows * size))
    axes = axes.reshape(n_rows, n_cols)

    image_shape = images_batch.shape[3:]
    heatmap_shape = heatmaps_batch.shape[3:]

    row_i = 0

    # images
    axes[row_i, 0].set_ylabel("image", size='large')

    images = image_batch_to_numpy(images_batch[batch_index])
    images = denormalize_image(images).astype(np.uint8)
    images = images[..., ::-1]  # bgr -> rgb

    # 2D keypoints (pred)
    if keypoints_2d_batch is not None:
        # axes[row_i, 0].set_ylabel("2d keypoints (pred)", size='large')

        keypoints_2d = to_numpy(keypoints_2d_batch)[batch_index]
        for view_i in range(n_cols):
            axes[row_i][view_i].imshow(images[view_i])
            keypoints_2d_gt_proj = project_3d_points_to_image_plane_without_distortion(
                proj_matricies_batch[batch_index,
                                     view_i].detach().cpu().numpy(),
                keypoints_3d_batch_gt[batch_index].detach().cpu().numpy())
            keypoints_2d_pred_proj = project_3d_points_to_image_plane_without_distortion(
                proj_matricies_batch[batch_index,
                                     view_i].detach().cpu().numpy(),
                keypoints_3d_batch_pred[batch_index].detach().cpu().numpy())
            draw_2d_keypoints(keypoints_2d_gt_proj,
                              axes[row_i][view_i],
                              kind=kind,
                              color="red",
                              label="gt")
            draw_2d_keypoints(keypoints_2d[view_i],
                              axes[row_i][view_i],
                              kind=kind,
                              color="green",
                              label="pred")
            draw_2d_keypoints(keypoints_2d_pred_proj,
                              axes[row_i][view_i],
                              kind=pred_kind,
                              color="blue",
                              label="proj")

    row_i += 1

    # cuboids
    if cuboids_batch is not None:
        axes[row_i, 0].set_ylabel("cuboid", size='large')

        for view_i in range(n_cols):
            cuboid = cuboids_batch[batch_index]
            axes[row_i][view_i].imshow(
                cuboid.render(
                    proj_matricies_batch[batch_index,
                                         view_i].detach().cpu().numpy(),
                    images[view_i].copy()))
        row_i += 1

    # confidences
    if confidences_batch is not None:
        axes[row_i, 0].set_ylabel("confidences", size='large')

        for view_i in range(n_cols):
            confidences = to_numpy(confidences_batch[batch_index, view_i])
            xs = np.arange(len(confidences))

            axes[row_i, view_i].bar(xs, confidences, color='green')
            axes[row_i, view_i].set_xticks(xs)
            if torch.max(confidences_batch).item() <= 1.0:
                axes[row_i, view_i].set_ylim(0.0, 1.0)

    axes[0][0].legend()
    # fig.tight_layout()
    fig_image = fig2data(fig)
    # print("fig:", fig_image.shape)
    plt.close('all')
    fig_image = fig_image[..., ::-1]

    return fig_image


def visualize_batch_cv2(images_batch,
                        heatmaps_batch,
                        keypoints_2d_batch,
                        proj_matricies_batch,
                        keypoints_3d_batch_gt,
                        keypoints_3d_batch_pred,
                        kind="cmu",
                        cuboids_batch=None,
                        confidences_batch=None,
                        batch_index=0,
                        size=5,
                        max_n_cols=10,
                        pred_kind=None,
                        keypoints_3d_batch_pred_1=None):
    if pred_kind is None:
        pred_kind = kind

    n_views, n_joints = images_batch.shape[1], images_batch.shape[2]

    n_rows = 3
    n_rows = n_rows + 1 if keypoints_2d_batch is not None else n_rows
    n_rows = n_rows + 1 if cuboids_batch is not None else n_rows
    n_rows = n_rows + 1 if confidences_batch is not None else n_rows
    n_rows = n_rows + 1 if keypoints_3d_batch_pred_1 is not None else n_rows

    n_cols = min(n_views, max_n_cols)
    fig, axes = plt.subplots(ncols=n_cols,
                             nrows=n_rows,
                             figsize=(n_cols * size, n_rows * size))
    axes = axes.reshape(n_rows, n_cols)

    row_i = 0

    # images
    axes[row_i, 0].set_ylabel("image", size='large')

    images = image_batch_to_numpy(images_batch[batch_index])
    images = denormalize_image(images).astype(np.uint8)
    images = images[..., ::-1]  # bgr -> rgb

    for view_i in range(n_cols):
        axes[row_i][view_i].imshow(images[view_i])
    row_i += 1

    # 2D keypoints (pred)
    if keypoints_2d_batch is not None:
        axes[row_i, 0].set_ylabel("2d keypoints (pred)", size='large')

        keypoints_2d = to_numpy(keypoints_2d_batch)[batch_index]
        for view_i in range(n_cols):
            axes[row_i][view_i].imshow(
                np.ones((images[view_i].shape[0], images[view_i].shape[1],
                         images[view_i].shape[2])))
            # axes[row_i][view_i].imshow(images[view_i])
            # keypoints_2d_gt_proj = project_3d_points_to_image_plane_without_distortion(proj_matricies_batch[batch_index, view_i].detach().cpu().numpy(), keypoints_3d_batch_gt[batch_index].detach().cpu().numpy())
            # draw_2d_pose(keypoints_2d_gt_proj, axes[row_i][view_i], kind=kind, color_flag='white')
            draw_2d_pose(keypoints_2d[view_i], axes[row_i][view_i], kind=kind)

        row_i += 1

    # 2D keypoints (gt projected)
    axes[row_i, 0].set_ylabel("2d keypoints (gt projected)", size='large')

    for view_i in range(n_cols):
        axes[row_i][view_i].imshow(images[view_i])
        keypoints_2d_gt_proj = project_3d_points_to_image_plane_without_distortion(
            proj_matricies_batch[batch_index, view_i].detach().cpu().numpy(),
            keypoints_3d_batch_gt[batch_index].detach().cpu().numpy())
        draw_2d_pose(keypoints_2d_gt_proj, axes[row_i][view_i], kind=kind)
    row_i += 1

    # 2D keypoints (pred projected)
    axes[row_i, 0].set_ylabel("2d keypoints (pred projected)", size='large')

    for view_i in range(n_cols):
        axes[row_i][view_i].imshow(images[view_i])
        keypoints_2d_pred_proj = project_3d_points_to_image_plane_without_distortion(
            proj_matricies_batch[batch_index, view_i].detach().cpu().numpy(),
            keypoints_3d_batch_pred[batch_index].detach().cpu().numpy())
        draw_2d_pose(keypoints_2d_pred_proj,
                     axes[row_i][view_i],
                     kind=pred_kind)
    row_i += 1

    # 2D keypoints (pred projected 1)
    if keypoints_3d_batch_pred_1 is not None:
        axes[row_i, 0].set_ylabel("2d keypoints (pred_1 projected)",
                                  size='large')

        for view_i in range(n_cols):
            axes[row_i][view_i].imshow(images[view_i])
            keypoints_2d_pred_proj_1 = project_3d_points_to_image_plane_without_distortion(
                proj_matricies_batch[batch_index,
                                     view_i].detach().cpu().numpy(),
                keypoints_3d_batch_pred_1[batch_index].detach().cpu().numpy())
            draw_2d_pose(keypoints_2d_pred_proj_1,
                         axes[row_i][view_i],
                         kind=pred_kind)
        row_i += 1
    # cuboids
    if cuboids_batch is not None:
        axes[row_i, 0].set_ylabel("cuboid", size='large')

        for view_i in range(n_cols):
            cuboid = cuboids_batch[batch_index]
            axes[row_i][view_i].imshow(
                cuboid.render(
                    proj_matricies_batch[batch_index,
                                         view_i].detach().cpu().numpy(),
                    images[view_i].copy()))
        row_i += 1

    # confidences
    if confidences_batch is not None:
        axes[row_i, 0].set_ylabel("confidences", size='large')

        for view_i in range(n_cols):
            confidences = to_numpy(confidences_batch[batch_index, view_i])
            xs = np.arange(len(confidences))

            axes[row_i, view_i].bar(xs, confidences, color='green')
            axes[row_i, view_i].set_xticks(xs)
            if torch.max(confidences_batch).item() <= 1.0:
                axes[row_i, view_i].set_ylim(0.0, 1.0)

    fig_image = fig2data(fig)
    plt.close('all')
    fig_image = fig_image[..., ::-1]

    return fig_image


def visualize_batch(images_batch,
                    heatmaps_batch,
                    keypoints_2d_batch,
                    proj_matricies_batch,
                    keypoints_3d_batch_gt,
                    keypoints_3d_batch_pred,
                    kind="cmu",
                    cuboids_batch=None,
                    confidences_batch=None,
                    batch_index=0,
                    size=5,
                    max_n_cols=10,
                    pred_kind=None,
                    keypoints_3d_batch_pred_1=None):
    if pred_kind is None:
        pred_kind = kind

    n_views, n_joints = heatmaps_batch.shape[1], heatmaps_batch.shape[2]

    n_rows = 3
    n_rows = n_rows + 1 if keypoints_2d_batch is not None else n_rows
    n_rows = n_rows + 1 if cuboids_batch is not None else n_rows
    n_rows = n_rows + 1 if confidences_batch is not None else n_rows
    n_rows = n_rows + 1 if keypoints_3d_batch_pred_1 is not None else n_rows

    n_cols = min(n_views, max_n_cols)
    fig, axes = plt.subplots(ncols=n_cols,
                             nrows=n_rows,
                             figsize=(n_cols * size, n_rows * size))
    axes = axes.reshape(n_rows, n_cols)

    image_shape = images_batch.shape[3:]
    heatmap_shape = heatmaps_batch.shape[3:]

    row_i = 0

    # images
    axes[row_i, 0].set_ylabel("image", size='large')

    images = image_batch_to_numpy(images_batch[batch_index])
    images = denormalize_image(images).astype(np.uint8)
    images = images[..., ::-1]  # bgr -> rgb

    for view_i in range(n_cols):
        axes[row_i][view_i].imshow(images[view_i])
    row_i += 1

    # 2D keypoints (pred)
    if keypoints_2d_batch is not None:
        axes[row_i, 0].set_ylabel("2d keypoints (pred)", size='large')

        keypoints_2d = to_numpy(keypoints_2d_batch)[batch_index]
        for view_i in range(n_cols):
            axes[row_i][view_i].imshow(images[view_i])
            draw_2d_pose(keypoints_2d[view_i], axes[row_i][view_i], kind=kind)
        row_i += 1

    # 2D keypoints (gt projected)
    axes[row_i, 0].set_ylabel("2d keypoints (gt projected)", size='large')

    for view_i in range(n_cols):
        axes[row_i][view_i].imshow(images[view_i])
        keypoints_2d_gt_proj = project_3d_points_to_image_plane_without_distortion(
            proj_matricies_batch[batch_index, view_i].detach().cpu().numpy(),
            keypoints_3d_batch_gt[batch_index].detach().cpu().numpy())
        draw_2d_pose(keypoints_2d_gt_proj, axes[row_i][view_i], kind=kind)
    row_i += 1

    # 2D keypoints (pred projected)
    axes[row_i, 0].set_ylabel("2d keypoints (pred projected)", size='large')

    for view_i in range(n_cols):
        axes[row_i][view_i].imshow(images[view_i])
        keypoints_2d_pred_proj = project_3d_points_to_image_plane_without_distortion(
            proj_matricies_batch[batch_index, view_i].detach().cpu().numpy(),
            keypoints_3d_batch_pred[batch_index].detach().cpu().numpy())
        draw_2d_pose(keypoints_2d_pred_proj,
                     axes[row_i][view_i],
                     kind=pred_kind)
    row_i += 1

    # 2D keypoints (pred projected 1)
    if keypoints_3d_batch_pred_1 is not None:
        axes[row_i, 0].set_ylabel("2d keypoints (pred_1 projected)",
                                  size='large')

        for view_i in range(n_cols):
            axes[row_i][view_i].imshow(images[view_i])
            keypoints_2d_pred_proj_1 = project_3d_points_to_image_plane_without_distortion(
                proj_matricies_batch[batch_index,
                                     view_i].detach().cpu().numpy(),
                keypoints_3d_batch_pred_1[batch_index].detach().cpu().numpy())
            draw_2d_pose(keypoints_2d_pred_proj_1,
                         axes[row_i][view_i],
                         kind=pred_kind)
        row_i += 1
    # cuboids
    if cuboids_batch is not None:
        axes[row_i, 0].set_ylabel("cuboid", size='large')

        for view_i in range(n_cols):
            cuboid = cuboids_batch[batch_index]
            axes[row_i][view_i].imshow(
                cuboid.render(
                    proj_matricies_batch[batch_index,
                                         view_i].detach().cpu().numpy(),
                    images[view_i].copy()))
        row_i += 1

    # confidences
    if confidences_batch is not None:
        axes[row_i, 0].set_ylabel("confidences", size='large')

        for view_i in range(n_cols):
            confidences = to_numpy(confidences_batch[batch_index, view_i])
            xs = np.arange(len(confidences))

            axes[row_i, view_i].bar(xs, confidences, color='green')
            axes[row_i, view_i].set_xticks(xs)
            if torch.max(confidences_batch).item() <= 1.0:
                axes[row_i, view_i].set_ylim(0.0, 1.0)

    fig.tight_layout()

    fig_image = fig_to_array(fig)

    plt.close('all')

    return fig_image


def visualize_heatmaps(images_batch,
                       heatmaps_batch,
                       kind="cmu",
                       batch_index=0,
                       size=5,
                       max_n_rows=10,
                       max_n_cols=10,
                       view_num=None):
    n_views, n_joints = heatmaps_batch.shape[1], heatmaps_batch.shape[2]
    heatmap_shape = heatmaps_batch.shape[3:]

    n_cols, n_rows = min(n_joints + 1, max_n_cols), min(n_views, max_n_rows)
    fig, axes = plt.subplots(ncols=n_cols,
                             nrows=n_rows,
                             figsize=(n_cols * size, n_rows * size))
    axes = axes.reshape(n_rows, n_cols)

    # images
    # print("images_shape:", images_batch[batch_index].shape)
    if view_num is not None:
        images = image_batch_to_numpy(images_batch[batch_index,
                                                   view_num].expand(
                                                       4, -1, -1, -1))
        
    else:
        images = image_batch_to_numpy(images_batch[batch_index])
    images = denormalize_image(images).astype(np.uint8)
    images = images[..., ::-1]  # bgr ->

    # heatmaps
    heatmaps = to_numpy(heatmaps_batch[batch_index])

    for row in range(n_rows):
        for col in range(n_cols):
            if col == 0:
                axes[row, col].set_ylabel(str(row), size='large')
                axes[row, col].imshow(images[row])
            else:
                if row == 0:
                    joint_name = JOINT_NAMES_DICT[kind][
                        col - 1] if kind in JOINT_NAMES_DICT else str(col - 1)
                    axes[row, col].set_title(joint_name)

                axes[row, col].imshow(resize_image(images[row], heatmap_shape))
                axes[row, col].imshow(heatmaps[row, col - 1], alpha=0.5)

    fig.tight_layout()

    # fig_image = fig_to_array(fig)
    fig_image = fig2data(fig)
    fig_image = fig_image[..., ::-1]
    plt.close('all')

    return fig_image

def visualize_heatmaps_T(images_batch,
                       heatmaps_batch,
                       kind="cmu",
                       batch_index=0,
                       size=5,
                       max_n_rows=10,
                       max_n_cols=10,
                       view_num=None):
    heatmaps_batch = heatmaps_batch.permute(0, 2, 1, 3, 4)
    n_views, n_joints = heatmaps_batch.shape[1], heatmaps_batch.shape[2]
    heatmap_shape = heatmaps_batch.shape[3:]

    n_cols, n_rows = min(n_joints + 1, max_n_cols), min(n_views, max_n_rows)
    fig, axes = plt.subplots(ncols=n_cols,
                             nrows=n_rows,
                             figsize=(n_cols * size, n_rows * size))
    axes = axes.reshape(n_rows, n_cols)

    # images
    # print("images_shape:", images_batch[batch_index].shape)
    if view_num is not None:
        images = image_batch_to_numpy(images_batch[batch_index,
                                                   view_num].expand(
                                                       4, -1, -1, -1))
        
    else:
        images = image_batch_to_numpy(images_batch[batch_index])
    images = denormalize_image(images).astype(np.uint8)
    images = images[..., ::-1]  # bgr ->

    # heatmaps
    heatmaps = to_numpy(heatmaps_batch[batch_index])

    for row in range(n_rows):
        for col in range(n_cols):
            if col == 0:
                axes[row, col].set_ylabel(str(row), size='large')
                axes[row, col].imshow(images[row])
            else:
                if row == 0:
                    joint_name = JOINT_NAMES_DICT[kind][
                        col - 1] if kind in JOINT_NAMES_DICT else str(col - 1)
                    axes[row, col].set_title(joint_name)

                axes[row, col].imshow(resize_image(images[row], heatmap_shape))
                axes[row, col].imshow(heatmaps[row, col - 1], alpha=0.5)

    fig.tight_layout()

    # fig_image = fig_to_array(fig)
    fig_image = fig2data(fig)
    fig_image = fig_image[..., ::-1]
    plt.close('all')

    return fig_image

def visualize_volumes(images_batch,
                      volumes_batch,
                      proj_matricies_batch,
                      kind="cmu",
                      cuboids_batch=None,
                      batch_index=0,
                      size=5,
                      max_n_rows=10,
                      max_n_cols=10):
    n_views, n_joints = volumes_batch.shape[1], volumes_batch.shape[2]

    n_cols, n_rows = min(n_joints + 1, max_n_cols), min(n_views, max_n_rows)
    fig = plt.figure(figsize=(n_cols * size, n_rows * size))

    # images
    images = image_batch_to_numpy(images_batch[batch_index])
    images = denormalize_image(images).astype(np.uint8)
    images = images[..., ::-1]  # bgr ->

    # heatmaps
    volumes = to_numpy(volumes_batch[batch_index])

    for row in range(n_rows):
        for col in range(n_cols):
            if col == 0:
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
                ax.set_ylabel(str(row), size='large')

                cuboid = cuboids_batch[batch_index]
                ax.imshow(
                    cuboid.render(
                        proj_matricies_batch[batch_index,
                                             row].detach().cpu().numpy(),
                        images[row].copy()))
            else:
                ax = fig.add_subplot(n_rows,
                                     n_cols,
                                     row * n_cols + col + 1,
                                     projection='3d')

                if row == 0:
                    joint_name = JOINT_NAMES_DICT[kind][
                        col - 1] if kind in JOINT_NAMES_DICT else str(col - 1)
                    ax.set_title(joint_name)

                draw_voxels(volumes[col - 1], ax, norm=True)

    fig.tight_layout()

    fig_image = fig2data(fig)
    fig_image = fig_image[..., ::-1]

    plt.close('all')

    return fig_image


def draw_2d_pose(keypoints,
                 ax,
                 kind='cmu',
                 keypoints_mask=None,
                 point_size=64,
                 line_width=3,
                 radius=None,
                 color_flag=None):
    """
    Visualizes a 2d skeleton

    Args
        keypoints numpy array of shape (19, 2): pose to draw in CMU format.
        ax: matplotlib axis to draw on
    """
    connectivity = CONNECTIVITY_DICT[kind]

    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)

    # connections
    for (index_from, index_to) in connectivity:
        if index_from == 0 or index_from == 1 or index_from == 2 or index_from == 10 or index_from == 11 or index_to == 12:
            color = 'white'
        elif index_from == 3 or index_from == 4 or index_from == 5 or index_to == 13 or index_from == 13 or index_from == 14:
            color = 'white'
        else:
            color = 'white'
        color = np.array([150, 150, 150]) / 255
        if color_flag is not None:
            color = np.array([200, 200, 200]) / 255
        if keypoints_mask[index_from] and keypoints_mask[index_to]:
            xs, ys = [
                np.array([keypoints[index_from, j], keypoints[index_to, j]])
                for j in range(2)
            ]
            ax.plot(xs, ys, c=color, lw=line_width, zorder=1)

    # points
    for index in range(17):
        point_color = 'black'
        if index in [0, 1, 2, 10, 11, 12]:
            point_color = np.array([181, 93, 96]) / 255
        elif index in [3, 4, 5, 13, 14, 15]:
            point_color = np.array([89, 117, 164]) / 255
        if color_flag is not None:
            point_color = np.array([200, 200, 200]) / 255
        # if index in [0,1,2,10,11,12]:
        #     point_color = 'red'
        # elif index in [3,4,5,13,14,15]:
        #     point_color = 'green'
        if color_flag is not None:
            ax.scatter(keypoints[keypoints_mask][index, 0],
                       keypoints[keypoints_mask][index, 1],
                       c=point_color,
                       s=point_size,
                       zorder=2)
        else:
            ax.scatter(keypoints[keypoints_mask][index, 0],
                       keypoints[keypoints_mask][index, 1],
                       c=point_color,
                       s=point_size,
                       linewidths=line_width,
                       edgecolors=color,
                       zorder=2)

    if radius is not None:
        root_keypoint_index = 0
        xroot, yroot = keypoints[root_keypoint_index,
                                 0], keypoints[root_keypoint_index, 1]

        ax.set_xlim([-radius + xroot, radius + xroot])
        ax.set_ylim([-radius + yroot, radius + yroot])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


def draw_2d_keypoints(keypoints,
                      ax,
                      kind='cmu',
                      keypoints_mask=None,
                      point_size=2,
                      line_width=1,
                      radius=None,
                      color=None,
                      label=None):
    """
    Visualizes a 2d keypoints

    Args
        keypoints numpy array of shape (19, 2): pose to draw in CMU format.
        ax: matplotlib axis to draw on
    """
    connectivity = CONNECTIVITY_DICT[kind]

    color = 'blue' if color is None else color
    # print("color:{},label:{}".format(color, label))
    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)

    # points
    ax.scatter(keypoints[keypoints_mask][:, 0],
               keypoints[keypoints_mask][:, 1],
               c=color,
               s=point_size,
               label=label)

    # connections
    # for (index_from, index_to) in connectivity:
    #     if keypoints_mask[index_from] and keypoints_mask[index_to]:
    #         xs, ys = [np.array([keypoints[index_from, j], keypoints[index_to, j]]) for j in range(2)]
    #         ax.plot(xs, ys, c=color, lw=line_width)

    if radius is not None:
        root_keypoint_index = 0
        xroot, yroot = keypoints[root_keypoint_index,
                                 0], keypoints[root_keypoint_index, 1]

        ax.set_xlim([-radius + xroot, radius + xroot])
        ax.set_ylim([-radius + yroot, radius + yroot])

    ax.set_aspect('equal')


def draw_2d_pose_cv2(keypoints,
                     canvas,
                     kind='cmu',
                     keypoints_mask=None,
                     point_size=2,
                     point_color=(255, 255, 255),
                     line_width=1,
                     radius=None,
                     color=None,
                     anti_aliasing_scale=1):
    canvas = canvas.copy()

    shape = np.array(canvas.shape[:2])
    new_shape = shape * anti_aliasing_scale
    canvas = resize_image(canvas, tuple(new_shape))

    keypoints = keypoints * anti_aliasing_scale
    point_size = point_size * anti_aliasing_scale
    line_width = line_width * anti_aliasing_scale

    connectivity = CONNECTIVITY_DICT[kind]

    color = 'blue' if color is None else color

    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)

    # connections
    for i, (index_from, index_to) in enumerate(connectivity):
        if keypoints_mask[index_from] and keypoints_mask[index_to]:
            pt_from = tuple(np.array(keypoints[index_from, :]).astype(int))
            pt_to = tuple(np.array(keypoints[index_to, :]).astype(int))

            if kind in COLOR_DICT:
                color = COLOR_DICT[kind][i]
            else:
                color = (0, 0, 255)

            cv2.line(canvas, pt_from, pt_to, color=color, thickness=line_width)

    if kind == 'coco':
        mid_collarbone = (keypoints[5, :] + keypoints[6, :]) / 2
        nose = keypoints[0, :]

        pt_from = tuple(np.array(nose).astype(int))
        pt_to = tuple(np.array(mid_collarbone).astype(int))

        if kind in COLOR_DICT:
            color = (153, 0, 51)
        else:
            color = (0, 0, 255)

        cv2.line(canvas, pt_from, pt_to, color=color, thickness=line_width)

    # points
    for pt in keypoints[keypoints_mask]:
        cv2.circle(canvas,
                   tuple(pt.astype(int)),
                   point_size,
                   color=point_color,
                   thickness=-1)

    canvas = resize_image(canvas, tuple(shape))

    return canvas


def draw_3d_pose(keypoints,
                 ax,
                 keypoints_mask=None,
                 kind='cmu',
                 radius=None,
                 root=None,
                 point_size=49,
                 line_width=2,
                 draw_connections=True):
    connectivity = CONNECTIVITY_DICT[kind]

    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)

    if draw_connections:
        # Make connection matrix
        for i, joint in enumerate(connectivity):
            if keypoints_mask[joint[0]] and keypoints_mask[joint[1]]:
                xs, ys, zs = [
                    np.array([keypoints[joint[0], j], keypoints[joint[1], j]])
                    for j in range(3)
                ]

                if kind in COLOR_DICT:
                    color = COLOR_DICT[kind][i]
                else:
                    color = (0, 0, 255)

                color = np.array(color) / 255

                # black bone
                color = np.array([90, 90, 90]) / 255
                ax.plot(xs, ys, zs, lw=line_width, c=color)

        if kind == 'coco':
            mid_collarbone = (keypoints[5, :] + keypoints[6, :]) / 2
            nose = keypoints[0, :]

            xs, ys, zs = [
                np.array([nose[j], mid_collarbone[j]]) for j in range(3)
            ]

            if kind in COLOR_DICT:
                color = (153, 0, 51)
            else:
                color = (0, 0, 255)

            color = np.array(color) / 255

            ax.plot(xs, ys, zs, lw=line_width, c=color)

    for index in range(17):
        point_color = np.array([90, 90, 90]) / 255
        if index in [0, 1, 2, 10, 11, 12]:
            point_color = np.array([181, 93, 96]) / 255
        elif index in [3, 4, 5, 13, 14, 15]:
            point_color = np.array([89, 117, 164]) / 255
        # ax.scatter(keypoints[keypoints_mask][index, 0], keypoints[keypoints_mask][index, 1], keypoints[keypoints_mask][index, 2],
        #             s=point_size, c=point_color, linewidths=line_width, edgecolors=np.array([70,70,70])/255)  # np.array([230, 145, 56])/255
        ax.scatter(keypoints[keypoints_mask][index, 0],
                   keypoints[keypoints_mask][index, 1],
                   keypoints[keypoints_mask][index, 2],
                   s=point_size,
                   c=point_color)

    if radius is not None:
        if root is None:
            root = np.mean(keypoints, axis=0)
        xroot, yroot, zroot = root
        ax.set_xlim([-radius + xroot, radius + xroot])
        ax.set_ylim([-radius + yroot, radius + yroot])
        ax.set_zlim([-radius + zroot, radius + zroot])

    # ax.set_aspect('equal')

    # Get rid of the panes
    background_color = np.array([252, 252, 252]) / 255

    ax.w_xaxis.set_pane_color(background_color)
    ax.w_yaxis.set_pane_color(background_color)
    ax.w_zaxis.set_pane_color(background_color)

    # Get rid of the ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def draw_voxels(voxels, ax, shape=(8, 8, 8), norm=True, alpha=0.1):
    # resize for visualization
    zoom = np.array(shape) / np.array(voxels.shape)
    voxels = skimage.transform.resize(voxels,
                                      shape,
                                      mode='constant',
                                      anti_aliasing=True)
    voxels = voxels.transpose(2, 0, 1)

    if norm and voxels.max() - voxels.min() > 0:
        voxels = (voxels - voxels.min()) / (voxels.max() - voxels.min())

    filled = np.ones(voxels.shape)

    # facecolors
    cmap = plt.get_cmap("Blues")

    facecolors_a = cmap(voxels, alpha=alpha)
    facecolors_a = facecolors_a.reshape(-1, 4)

    facecolors_hex = np.array(
        list(
            map(lambda x: matplotlib.colors.to_hex(x, keep_alpha=True),
                facecolors_a)))
    facecolors_hex = facecolors_hex.reshape(*voxels.shape)

    # explode voxels to perform 3d alpha rendering (https://matplotlib.org/devdocs/gallery/mplot3d/voxels_numpy_logo.html)
    def explode(data):
        size = np.array(data.shape) * 2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e

    filled_2 = explode(filled)
    facecolors_2 = explode(facecolors_hex)

    # shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    # draw voxels
    ax.voxels(x, y, z, filled_2, facecolors=facecolors_2)

    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.invert_xaxis()
    ax.invert_zaxis()


def point3d(ax, theta1, theta2, loc, color, label_name=None):
    skeleton_index_human36 = [[0, 1, 2, 6], [6, 3, 4, 5], [6, 7, 8, 9, 16],
                              [10, 11, 12, 8], [8, 13, 14, 15]]

    # 绘制 3D 散点
    x = loc[:, 0]
    y = loc[:, 2]
    z = loc[:, 1]
    # points

    for index in range(17):
        # 区分左右
        
        point_color = np.array([70, 70, 70]) / 255
        if index in [0, 1, 2, 10, 11, 12]:
            point_color = np.array([181, 93, 96]) / 255
        elif index in [3, 4, 5, 13, 14, 15]:
            point_color = np.array([89, 117, 164]) / 255
        # points = ax.scatter(xs=x[index],    # x 轴坐标
        #                     ys=z[index],    # y 轴坐标
        #                     zs=y[index],    # z 轴坐标
        #                     zdir='z',    #
        #                     c= point_color,    # color
        #                     s=25,    # size
        #                     linewidths=2,
        #                     edgecolors=np.array([70,70,70])/255,
        #                     label=label_name,
        #                     zorder=2
        #                     )

        # 区分真值和预测
        if label_name=="gt":
            point_color = np.array([200, 200, 200]) / 255
        # if label_name is not None:
        #     point_color=color
        points = ax.scatter(
            xs=x[index],  # x 轴坐标
            ys=z[index],  # y 轴坐标
            zs=y[index],  # z 轴坐标
            zdir='z',  #
            c=point_color,  # color
            s=25,  # size
            label=label_name,
            zorder=2)

    # 绘制骨骼连线
    for i in range(5):
        skeleton_index = skeleton_index_human36[i]
        part_index_num = len(skeleton_index)
        # 区分gt pred
        # skeleton_color = color
        # 区分左右
        skeleton_color = np.array([70, 70, 70]) / 255
        if label_name == 'gt':
            skeleton_color = np.array([200, 200, 200]) / 255
        # if i == 0 or i == 3:
        #     skeleton_color = 'green'
        # elif i == 1 or i == 4:
        #     skeleton_color = 'yellow'
        # if i == 2:
        #     id1, id2, id3, id4, id5 = skeleton_index_human36[i]
        # id1, id2, id3, id4 = skeleton_index_human36[i]
        for j in range(part_index_num - 1):
            id1 = skeleton_index[j]
            id2 = skeleton_index[j + 1]
            ax.plot((x[id1], x[id2]), (z[id1], z[id2]), (y[id1], y[id2]),
                    color=skeleton_color,
                    lw=2,
                    zorder=1)

    # extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    # sz = extents[:, 1] - extents[:, 0]
    # centers = np.mean(extents, axis=1)
    # maxsize = max(abs(sz))
    # r = sz/2
    # axis_min = centers - r
    # axis_max = centers + r
    # ax.auto_scale_xyz(*np.column_stack((axis_min, axis_max)))
    # ax.xaxis.set_major_locator(MultipleLocator(200))
    # ax.yaxis.set_major_locator(MultipleLocator(200))
    # ax.zaxis.set_major_locator(MultipleLocator(200))
    x_range = max(abs(min(x)), abs(max(x)))
    z_range = max(abs(min(z)), abs(max(z)))
    ax.set_xlim(-x_range - 50, x_range + 50)
    ax.set_ylim(-z_range - 50, z_range + 50)
    ax.set_zlim(min(y), max(y) + 100)

    # 调整视角
    ax.view_init(
        elev=theta1,  # 仰角
        azim=theta2  # 方位角
    )


def draw_3d_pose_in_two_views(axes,
                              keypoints_3d_gt=None,
                              keypoints_3d_pred=None,
                              keypoints_3d_psm=None):
    #keypoints_3d: numpy with shape(1,njoints,3)
    # x_min = np.min(keypoints_3d_pred[:,:,0])
    # y_min = np.min(keypoints_3d_pred[:,:,2])
    # z_min = np.min(keypoints_3d_pred[:,:,1])
    # x_max = np.max(keypoints_3d_pred[:,:,0])
    # y_max = np.max(keypoints_3d_pred[:,:,2])
    # z_max = np.max(keypoints_3d_pred[:,:,1])
    paints = ['red', 'blue', 'green']

    ax0 = axes[0]
    ax0.set(
        xlabel='X',
        ylabel='Z',
        zlabel='Y',
        #    xlim=(-400, 500),
        #    zlim=(-800, 800),
        #    xticks=np.arange(-400, 500, 200),
        #    zticks=np.arange(-800, 800, 200),
    )
    # ax0.set_xticks([])
    # ax0.set_yticks([])
    # ax0.set_zticks([])
    background_color = np.array([50, 50, 50]) / 255
    background_color_white = np.array([255, 255, 255]) / 255
    # ax0.xaxis.set_major_locator(MultipleLocator(200))
    # ax0.yaxis.set_major_locator(MultipleLocator(200))
    # ax0.zaxis.set_major_locator(MultipleLocator(200))

    ax0.w_xaxis.set_pane_color(background_color_white)
    ax0.w_yaxis.set_pane_color(background_color_white)
    ax0.w_zaxis.set_pane_color(background_color)

    # Get rid of the ticks
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.set_zticklabels([])
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_zticks([])
    ax0.w_xaxis.line.set_lw(0.)
    ax0.w_yaxis.line.set_lw(0.)
    ax0.w_zaxis.line.set_lw(0.)
    ax0.grid(False)
    ax1 = axes[1]
    ax1.set(
        xlabel='X',
        ylabel='Z',
        zlabel='Y',
        # ylim=(-200, 200),
        # zlim=(-800, 800),
        # yticks=np.arange(-200, 200, 50),
        # zticks=np.arange(-800, 800, 200)
    )
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax1.set_zticks([])

    # ax1.xaxis.set_major_locator(MultipleLocator(200))
    # ax1.yaxis.set_major_locator(MultipleLocator(200))
    # ax1.zaxis.set_major_locator(MultipleLocator(200))
    ax1.w_xaxis.set_pane_color(background_color_white)
    ax1.w_yaxis.set_pane_color(background_color_white)
    ax1.w_zaxis.set_pane_color(background_color)

    # Get rid of the ticks
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.w_xaxis.line.set_lw(0.)
    ax1.w_yaxis.line.set_lw(0.)
    ax1.w_zaxis.line.set_lw(0.)
    ax1.grid(False)

    ax2 = axes[2]
    ax2.set(
        xlabel='X',
        ylabel='Z',
        zlabel='Y',
        # ylim=(-200, 200),
        # zlim=(-800, 800),
        # yticks=np.arange(-200, 200, 50),
        # zticks=np.arange(-800, 800, 200)
    )
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    # ax2.set_zticks([])

    # ax2.xaxis.set_major_locator(MultipleLocator(200))
    # ax2.yaxis.set_major_locator(MultipleLocator(200))
    # ax2.zaxis.set_major_locator(MultipleLocator(200))
    ax2.w_xaxis.set_pane_color(background_color_white)
    ax2.w_yaxis.set_pane_color(background_color_white)
    ax2.w_zaxis.set_pane_color(background_color)

    # Get rid of the ticks
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.w_xaxis.line.set_lw(0.)
    ax2.w_yaxis.line.set_lw(0.)
    ax2.w_zaxis.line.set_lw(0.)
    ax2.grid(False)

    ax3 = axes[3]
    ax3.set(
        xlabel='X',
        ylabel='Z',
        zlabel='Y',
        # ylim=(-200, 200),
        # zlim=(-800, 800),
        # yticks=np.arange(-200, 200, 50),
        # zticks=np.arange(-800, 800, 200)
    )
    # ax3.set_xticks([])
    # ax3.set_yticks([])
    # ax3.set_zticks([])

    # ax3.xaxis.set_major_locator(MultipleLocator(200))
    # ax3.yaxis.set_major_locator(MultipleLocator(200))
    # ax3.zaxis.set_major_locator(MultipleLocator(200))
    ax3.w_xaxis.set_pane_color(background_color_white)
    ax3.w_yaxis.set_pane_color(background_color_white)
    ax3.w_zaxis.set_pane_color(background_color)

    # Get rid of the ticks
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    ax3.w_xaxis.line.set_lw(0.)
    ax3.w_yaxis.line.set_lw(0.)
    ax3.w_zaxis.line.set_lw(0.)
    ax3.grid(False)

    if keypoints_3d_gt is not None:
        point3d(ax0, 0, 0, keypoints_3d_gt, paints[0], label_name="gt")
        point3d(ax1, 0, 20, keypoints_3d_gt, paints[0], label_name="gt")
        point3d(ax2, 0, 45, keypoints_3d_gt, paints[0], label_name="gt")
        point3d(ax3, 0, 90, keypoints_3d_gt, paints[0], label_name="gt")
    if keypoints_3d_pred is not None:
        point3d(ax0, 0, 0, keypoints_3d_pred, paints[1], label_name="pred")
        point3d(ax1, 0, 20, keypoints_3d_pred, paints[1], label_name="pred")
        point3d(ax2, 0, 45, keypoints_3d_pred, paints[1], label_name="pred")
        point3d(ax3, 0, 90, keypoints_3d_pred, paints[1], label_name="pred")
    if keypoints_3d_psm is not None:
        point3d(ax0, 0, 0, keypoints_3d_psm, paints[2], label_name="psm")
        point3d(ax1, 0, 20, keypoints_3d_psm, paints[2], label_name="psm")
        point3d(ax2, 0, 45, keypoints_3d_psm, paints[2], label_name="psm")
        point3d(ax3, 0, 90, keypoints_3d_psm, paints[2], label_name="psm")
    # plt.legend()