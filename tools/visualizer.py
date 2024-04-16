import argparse
from typing import Dict, List

import numpy as np
import open3d as o3d

# # Define label colors
# S3DIS_LABEL_COLORS = {0: [1, 0, 0],  # Red, ceiling
#                       1: [0, 1, 0],  # Green, floor
#                       2: [0, 0, 1],  # Blue, wall
#                       12: [0, 0, 0]  # Dark Blue, clutter
#                       }
#
# S3DIS_LABELS = {'ceiling': 0,
#                 'floor': 1,
#                 'wall': 2,
#                 'clutter': 12,
#                 }


# Define label colors
S3DIS_LABEL_COLORS = {
    0: [1, 0, 0],       # Red, ceiling
    1: [0, 1, 0],       # Green, floor
    2: [0, 0, 1],       # Blue, wall
    3: [1, 1, 0],       # Yellow, beam
    4: [1, 0, 1],       # Magenta, column
    5: [0, 1, 1],       # Cyan, window
    6: [0.5, 0.5, 0.5], # Gray, door
    7: [1, 0.5, 0],     # Orange, table
    8: [0, 0.5, 1],     # Sky blue, chair
    9: [0.5, 0, 1],     # Purple, sofa
    10: [0.5, 1, 0],    # Lime, bookcase
    11: [1, 0, 0.5],    # Pink, board
    12: [0, 0, 0]       # Dark Blue, clutter
}

S3DIS_LABELS = {
    'ceiling': 0,
    'floor': 1,
    'wall': 2,
    'beam': 3,
    'column': 4,
    'window': 5,
    'door': 6,
    'table': 7,
    'chair': 8,
    'sofa': 9,
    'bookcase': 10,
    'board': 11,
    'clutter': 12,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize point clouds with semantic segmentation predictions.")
    parser.add_argument("--pcd_filepath", type=str, help="Path to point cloud.")
    parser.add_argument("--pred_filepath", type=str, help="Path to prediction file.")

    args = parser.parse_args()

    return args


def visualize_point_cloud(pcd_filepath: str, pred_filepath: str) -> None:
    """
    Visualize point clouds with semantic segmentation predictions.

    Parameters
    ----------
    pcd_filepath : str
        Filepath of point cloud file.

    pred_filepath : str
        Filepath of prediction file.

    """

    def remove_points(vis, label):
        nonlocal preds

        label_indices = np.where(preds == label)[0]

        preds = np.delete(preds, label_indices, axis=0)

        pcd.points = o3d.utility.Vector3dVector(np.delete(np.asarray(pcd.points), label_indices, axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.delete(np.asarray(pcd.colors), label_indices, axis=0))

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()


    # Read the point cloud
    pcd = o3d.io.read_point_cloud(pcd_filepath)

    # Load predictions
    preds = np.load(pred_filepath)

    # Assign anything that's not ceiling, floor, or wall as clutter
    colors = [S3DIS_LABEL_COLORS[pred] if pred in S3DIS_LABEL_COLORS else S3DIS_LABEL_COLORS[12] for pred in preds]

    # Create segmented point cloud
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = pcd.points
    _pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    # Visualize the segmented point cloud
    o3d.visualization.draw_geometries([_pcd])

    # Visualize the point cloud
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)

    # Register key callbacks for removing points
    vis.register_key_callback(ord('F'), lambda vis: remove_points(vis, [S3DIS_LABELS['floor']]))
    vis.register_key_callback(ord('C'), lambda vis: remove_points(vis, [S3DIS_LABELS['ceiling']]))
    vis.register_key_callback(ord('W'), lambda vis: remove_points(vis, [S3DIS_LABELS['wall']]))

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    args = parse_args()

    # Visualize point cloud
    visualize_point_cloud(pcd_filepath=args.pcd_filepath, pred_filepath=args.pred_filepath)
