# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# class CameraPoseVisualizer:
#     def __init__(self, xlim, ylim, zlim):
#         self.fig = plt.figure(figsize=(8, 8))
#         self.ax = self.fig.add_subplot(projection = '3d')
#         self.ax.set_aspect("auto")
#         self.ax.set_xlim(xlim)
#         self.ax.set_ylim(ylim)
#         self.ax.set_zlim(zlim)
#         self.ax.set_xlabel('x')
#         self.ax.set_ylabel('y')
#         self.ax.set_zlabel('z')
#         self.ax.grid(False)
#         self.ax.set_facecolor('w')
#         self.ax.set_box_aspect([1, 1, 1])
#         print('initialize camera pose visualizer')

#     def extrinsic2pyramid(self, extrinsic, color='b', focal_len_scaled=5, aspect_ratio=0.3):
#         vertex_std = np.array([[0, 0, 0, 1],
#                                [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
#                                [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
#                                [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
#                                [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
#         vertex_std[:, 2] = -vertex_std[:, 2]  # 将 z 坐标反转 
#         vertex_transformed = vertex_std @ extrinsic.T
#         meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
#                             [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
#                             [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
#                             [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
#                             [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
#         self.ax.add_collection3d(
#             Poly3DCollection(meshes, facecolors=color, linewidths=0.6, edgecolors=color, alpha=0.2))

#     def customize_legend(self, list_label):
#         list_handle = []
#         for idx, label in enumerate(list_label):
#             color = plt.cm.rainbow(idx / len(list_label))
#             patch = Patch(color=color, label=label)
#             list_handle.append(patch)
#         plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

#     def colorbar(self, max_frame_length):
#         cmap = mpl.cm.rainbow
#         norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
#         self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

#     def show(self):
#         plt.title('Extrinsic Parameters')
#         plt.show()
        
#     def save(self, path):
#         # plt.title('Extrinsic Parameters')
#         plt.savefig(path, dpi=300)

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.widgets import Button, CheckButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.grid(False)
        self.ax.set_facecolor('w')
        self.ax.set_box_aspect([1, 1, 1])

    def extrinsic2pyramid(self, extrinsic, color='b', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_std[:, 2] = -vertex_std[:, 2]  # 将 z 坐标反转 
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                  [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.6, edgecolors=color, alpha=0.2))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()

    def save(self, path):
        plt.savefig(path, dpi=300)


class InteractiveCameraVisualizer:
    """Interactive 3D camera trajectory visualizer with GUI controls."""
    
    def __init__(self, c2ws, title="Camera Trajectory Visualizer"):
        self.c2ws = c2ws
        self.title = title
        
        # Calculate range
        if HAS_TORCH and isinstance(c2ws, torch.Tensor):
            c2ws_np = c2ws.cpu().numpy()
        else:
            c2ws_np = np.array(c2ws)
        self.rangesize = np.max(np.abs(c2ws_np[:, :3, 3])) * 1.1
        
        # Create figure with space for controls
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Adjust subplot to make room for controls
        self.fig.subplots_adjust(left=0.05, right=0.85, bottom=0.15, top=0.95)
        
        # Store colors - frame 0 = purple (rainbow(0)), last frame = red (rainbow(1))
        self.num_matrices = c2ws_np.shape[0]
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_matrices))
        self.c2ws_np = c2ws_np
        
        # Draw initial scene
        self._setup_axes()
        self._draw_cameras()
        self._add_trajectory_line()
        self._add_colorbar()
        self._add_controls()
        
    def _setup_axes(self):
        """Setup axis properties."""
        self.ax.set_xlim([-self.rangesize, self.rangesize])
        self.ax.set_ylim([-self.rangesize, self.rangesize])
        self.ax.set_zlim([-self.rangesize, self.rangesize])
        # Camera coordinate: +x: right, +y: up, +z: backward
        self.ax.set_xlabel('X', fontsize=10, fontweight='bold')
        self.ax.set_ylabel('Y', fontsize=10, fontweight='bold')
        self.ax.set_zlabel('Z', fontsize=10, fontweight='bold')
        self.ax.set_title(self.title, fontsize=12, fontweight='bold', pad=10)
        self.ax.set_facecolor('#f8f9fa')
        self.ax.set_box_aspect([1, 1, 1])
        
    def _draw_cameras(self):
        """Draw camera pyramids."""
        focal_len_scaled = self.rangesize / 4
        aspect_ratio = 0.3
        
        for i in range(self.num_matrices):
            color = self.colors[i]
            extrinsic = self.c2ws_np[i]
            
            vertex_std = np.array([
                [0, 0, 0, 1],
                [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]
            ])
            vertex_std[:, 2] = -vertex_std[:, 2]
            vertex_transformed = vertex_std @ extrinsic.T
            
            meshes = [
                [vertex_transformed[0, :-1], vertex_transformed[1, :-1], vertex_transformed[2, :-1]],
                [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]
            ]
            self.ax.add_collection3d(
                Poly3DCollection(meshes, facecolors=color, linewidths=0.6, edgecolors=color, alpha=0.25))
    
    def _add_trajectory_line(self):
        """Add line connecting camera positions."""
        positions = self.c2ws_np[:, :3, 3]
        
        self.ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], 
                       'k-', linewidth=1.5, alpha=0.6, label='Trajectory')
        
        # Add start and end markers matching colormap colors
        start_color = plt.cm.rainbow(0.0)  # Purple for frame 0
        end_color = plt.cm.rainbow(1.0)    # Red for last frame
        self.ax.scatter(*positions[0], color=start_color, s=100, marker='o', label='Start (Frame 0)', zorder=5)
        self.ax.scatter(*positions[-1], color=end_color, s=100, marker='s', label=f'End (Frame {len(positions)-1})', zorder=5)
    
    def _add_colorbar(self):
        """Add colorbar showing frame numbers."""
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=self.num_matrices - 1)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Frame Number', fontsize=10)
    
    def _add_controls(self):
        """Add interactive control buttons."""
        # View preset buttons - matching the original draw_json views
        ax_front = self.fig.add_axes([0.05, 0.02, 0.1, 0.04])
        ax_top = self.fig.add_axes([0.16, 0.02, 0.1, 0.04])
        ax_side = self.fig.add_axes([0.27, 0.02, 0.1, 0.04])
        ax_iso = self.fig.add_axes([0.38, 0.02, 0.1, 0.04])
        
        self.btn_front = Button(ax_front, 'Front', color='lightblue', hovercolor='deepskyblue')
        self.btn_top = Button(ax_top, 'Top', color='lightblue', hovercolor='deepskyblue')
        self.btn_side = Button(ax_side, 'Side', color='lightblue', hovercolor='deepskyblue')
        self.btn_iso = Button(ax_iso, 'Isometric', color='lightblue', hovercolor='deepskyblue')
        
        # Use the same view angles as the original draw_json function
        # Camera coord: +x: right, +y: up, +z: backward
        self.btn_front.on_clicked(lambda e: self._set_view(90, -90))   # Front view
        self.btn_top.on_clicked(lambda e: self._set_view(180, -90))    # Top view
        self.btn_side.on_clicked(lambda e: self._set_view(0, 0, roll=90))     # Side view, y, z,
        self.btn_iso.on_clicked(lambda e: self._set_view(30, -60))     # Isometric
        
        # Add legend
        self.ax.legend(loc='upper left', fontsize=8)
        
        # Instructions text
        self.fig.text(0.55, 0.02, 'Drag to rotate | Scroll to zoom | Right-drag to pan', 
                      fontsize=9, style='italic', color='gray')
    
    def _set_view(self, elev, azim, roll=0):
        """Set camera view angle."""
        self.ax.view_init(elev=elev, azim=azim, roll=roll)
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the interactive visualization."""
        plt.show()
    
    def save(self, path):
        """Save current view to file."""
        self.fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {path}")


def visualize_trajectory_interactive(c2ws, title="Camera Trajectory"):
    """
    Show only the interactive 3D visualization without saving images.
    
    Args:
        c2ws: Camera-to-world matrices (N, 4, 4)
        title: Window title
    """
    if HAS_TORCH and isinstance(c2ws, torch.Tensor):
        c2ws_np = c2ws.cpu().numpy()
    else:
        c2ws_np = np.array(c2ws)
    
    interactive_vis = InteractiveCameraVisualizer(c2ws_np, title=title)
    interactive_vis.show()

