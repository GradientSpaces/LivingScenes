import sys

sys.path.append("..")

import os
import platform

node = platform.node()
if node == "endurance":  # my station
    os.environ["PYOPENGL_PLATFORM"] = "egl"
else:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import os.path as osp
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import trimesh
import torch

from transforms3d.euler import euler2mat
import trimesh
import os
import imageio
import numpy as np
import time
from matplotlib import cm
from shapely.geometry import Polygon


def render(
    # draw pointclouds
    pcl_list=None,
    pcl_color_list=None,
    pcl_radius_list=None,
    pcl_fallback_colors=[0.9, 0.9, 0.9, 1.0],
    # draw meshes
    mesh_list=None,
    mesh_color_list=None,
    # draw lines
    line_list=None,
    lines_color_list=None,
    lines_color_fallback=[0.5, 0.5, 0.5, 1.0],
    # draw arrows
    arrow_tuples=None,  # (Nx3, Nx3)
    arrow_radius=0.01,
    arrow_colors=None,  # N,4
    arrow_head=False,
    # render config
    shape=(640, 640),
    light_intensity=1.0,
    light_vertical_angle=-np.pi/4,
    yfov=np.pi / 3.0,
    cam_angle_yaw=0.0,
    cam_angle_pitch=0.0,
    cam_angle_z=0.0,
    cam_dist=1.0,
    cam_height=0.0,
    perpoint_color_flag=False,
    **kargs,
):
    if not isinstance(cam_angle_yaw, list):
        cam_angle_yaw = [cam_angle_yaw]
    if not isinstance(cam_angle_pitch, list):
        cam_angle_pitch = [cam_angle_pitch]
    if not isinstance(cam_angle_z, list):
        cam_angle_z = [cam_angle_z]
    if not isinstance(cam_dist, list):
        cam_dist = [cam_dist]
    if not isinstance(cam_height, list):
        cam_height = [cam_height]
    N_cam_pose = len(cam_angle_yaw)
    assert len(cam_angle_pitch) == N_cam_pose
    assert len(cam_angle_z) == N_cam_pose
    assert len(cam_dist) == N_cam_pose
    assert len(cam_height) == N_cam_pose

    cam_pose_list = []
    for i in range(N_cam_pose):
        cam_pose = np.eye(4)
        R = euler2mat(cam_angle_yaw[i], -cam_angle_pitch[i], cam_angle_z[i], "ryxz")
        cam_pose[:3, :3] = R
        cam_pose[2, 3] += cam_dist[i]
        cam_pose[:3, 3:] = R @ cam_pose[:3, 3:]
        cam_pose[1, 3] += cam_height[i]
        cam_pose_list.append(cam_pose)

    renderer = pyrender.OffscreenRenderer(shape[0], shape[1])
    scene = pyrender.Scene()

    ######################################################################
    # dlight = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
    # T_dl = np.eye(4)
    # T_dl[:3, :3] = euler2mat(-np.pi / 2.0, 0.0, 0.0, "rxyz")
    # scene.add(dlight, pose=T_dl)

    # plight = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
    # T_pl = np.eye(4)
    # T_pl[1, 3] += light_y
    # scene.add(plight, pose=T_pl)
    ######################################################################
    
    
    # light from 4 direction above
    for rot in [0.0,np.pi/2, np.pi, 3*np.pi/2]:
        dlight = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
        T_dl = np.eye(4)
        T_dl[:3, :3] = euler2mat(light_vertical_angle, rot, 0.0, "sxyz")
        scene.add(dlight, pose=T_dl)

    if pcl_list is not None:
        # PCL [N,3]
        assert len(pcl_list) == len(pcl_radius_list)
        for i in range(len(pcl_list)):
            pcl = pcl_list[i]
            radius = pcl_radius_list[i]
            assert isinstance(radius, float)
            color = pcl_fallback_colors if pcl_color_list is None else pcl_color_list[i]
            if isinstance(color, torch.Tensor):
                color = color.numpy()
            if isinstance(color, np.ndarray) and color.ndim == 1:
                # color = cm.viridis(color)
                # color = cm.seismic(color)
                color = cm.cool(color)
            if radius <= 0.0:
                m = pyrender.Mesh.from_points(pcl, colors=color)
                scene.add(m)
            else:
                sm = trimesh.creation.uv_sphere(radius=radius)
                if perpoint_color_flag and isinstance(color, np.ndarray):
                    for i in tqdm(range(len(pcl))):
                        sm = trimesh.creation.uv_sphere(radius=radius)
                        sm.visual.vertex_colors = color[i]
                        tfs = np.eye(4)
                        tfs[:3, 3] = pcl[i]
                        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                        scene.add(m)
                else:
                    sm.visual.vertex_colors = color
                    tfs = np.tile(np.eye(4), (pcl.shape[0], 1, 1))
                    tfs[:, :3, 3] = pcl
                    m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                    scene.add(m)
    if mesh_list is not None:
        for i in range(len(mesh_list)):
            if len(mesh_list[i].vertices) == 0:
                continue
            if mesh_color_list is not None and mesh_color_list[i] is not None:
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    roughnessFactor=0.0,
                    alphaMode="BLEND",
                    baseColorFactor=mesh_color_list[i],
                )
            else:
                material = None
            scene.add(pyrender.Mesh.from_trimesh(mesh_list[i], material=material))
    if line_list is not None:
        if lines_color_list is not None:
            assert len(lines_color_list) == len(line_list)
        else:
            lines_color_list = [lines_color_fallback] * len(line_list)
        for color, start_end in zip(lines_color_list, line_list):
            lines = np.hstack(start_end)
            lines = lines.reshape(-1, 3)
            line_color = np.asarray(color)
            if line_color.ndim == 1:
                line_color = np.tile(line_color, (len(lines), 1))
            else:
                assert len(line_color) * 2 == len(lines)
                line_color = np.hstack((line_color, line_color)).reshape(-1, 4)
            primitive = [pyrender.Primitive(lines, mode=1, color_0=line_color)]
            primitive_mesh = pyrender.Mesh(primitive)
            scene.add(primitive_mesh)
    if arrow_tuples is not None:
        # ! Note that this does not support list like the line
        arrow_start, arrow_dir = arrow_tuples
        assert len(arrow_start) == len(arrow_colors)
        for aid in range(len(arrow_start)):
            plot_arrow(
                scene,
                arrow_start[aid],
                arrow_dir[aid],
                tube_radius=arrow_radius,
                color=arrow_colors[aid],
                smooth=True,
                use_head=arrow_head,
            )
    rgb_list = []
    for cam_pose in cam_pose_list:
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=shape[0] / shape[1])
        camera = scene.add(camera, pose=cam_pose)
        rgb, _ = renderer.render(scene)
        scene.remove_node(camera)
        rgb_list.append(rgb)
    rgb_list = np.concatenate(rgb_list, 1)
    return rgb_list


def plot_arrow(
    scene,
    start_point,
    direction,
    tube_radius=0.01,
    color=(0.5, 0.5, 0.5),
    material=None,
    smooth=True,
    use_head=True,
):
    """Plot an arrow with start and end points.
    Parameters
    ----------
    start_point : (3,) float
        Origin point for the arrow
    direction : (3,) float
        Vector defining the arrow
    tube_radius : float
        Radius of plotted x,y,z axes.
    color : (3,) float
        The color of the tube.
    material:
        Material of mesh
    n_components : int
        The number of edges in each polygon representing the tube.
    smooth : bool
        If true, the mesh is smoothed before rendering.
    """
    end_point = start_point + direction
    if use_head:
        arrow_head = create_arrow_head(length=np.linalg.norm(direction), tube_radius=tube_radius)
        arrow_head_rot = trimesh.geometry.align_vectors(np.array([0, 0, 1]), direction)
        arrow_head_tf = np.matmul(
            trimesh.transformations.translation_matrix(end_point), arrow_head_rot
        )

    vec = np.array([start_point, end_point])

    plot3d_tube(scene, vec, tube_radius=tube_radius, color=color)
    if use_head:
        add_mesh(
            scene,
            arrow_head,
            T_mesh_world=arrow_head_tf,
            color=color,
            material=material,
            smooth=smooth,
        )


def add_mesh(
    scene,
    mesh,
    name=None,
    T_mesh_world=None,
    style="surface",
    color=(0.5, 0.5, 0.5),
    material=None,
    smooth=False,
):
    """Visualize a 3D triangular mesh.
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to visualize.
    name : str
        A name for the object to be added.
    T_mesh_world : autolab_core.RigidTransform
        The pose of the mesh, specified as a transformation from mesh frame to world frame.
    style : str
        Triangular mesh style, either 'surface' or 'wireframe'.
    color : 3-tuple
        Color tuple.
    material:
        Material of mesh
    smooth : bool
        If true, the mesh is smoothed before rendering.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Must provide a trimesh.Trimesh object")

    n = _create_node_from_mesh(
        mesh,
        name=name,
        pose=T_mesh_world,
        color=color,
        material=material,
        poses=None,
        wireframe=(style == "wireframe"),
        smooth=smooth,
    )
    scene.add_node(n)
    return n


def _create_node_from_mesh(
    mesh, name=None, pose=None, color=None, material=None, poses=None, wireframe=False, smooth=True
):
    """Helper method that creates a pyrender.Node from a trimesh.Trimesh"""
    # Create default pose
    if pose is None:
        pose = np.eye(4)

    # Create vertex colors if needed
    if color is not None:
        color = np.asanyarray(color, dtype=np.float)
        if color.ndim == 1 or len(color) != len(mesh.vertices):
            color = np.repeat(color[np.newaxis, :], len(mesh.vertices), axis=0)
        mesh.visual.vertex_colors = color

    if material is None and mesh.visual.kind != "texture":
        if color is not None:
            material = None
        else:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
                metallicFactor=0.2,
                roughnessFactor=0.8,
            )

    m = pyrender.Mesh.from_trimesh(
        mesh, material=material, poses=poses, wireframe=wireframe, smooth=smooth
    )
    return pyrender.Node(mesh=m, name=name, matrix=pose)


import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def plot3d_tube(
    scene,
    points,
    tube_radius=None,
    name=None,
    pose=None,
    color=(0.5, 0.5, 0.5),
    material=None,
    n_components=10,
    smooth=True,
):
    """Plot a 3d curve through a set of points using tubes.
    Parameters
    ----------
    points : (n,3) float
        A series of 3D points that define a curve in space.
    tube_radius : float
        Radius of tube representing curve.
    name : str
        A name for the object to be added.
    pose : autolab_core.RigidTransform
        Pose of object relative to world.
    color : (3,) float
        The color of the tube.
    material:
        Material of mesh
    n_components : int
        The number of edges in each polygon representing the tube.
    smooth : bool
        If true, the mesh is smoothed before rendering.
    """
    # Generate circular polygon
    vec = np.array([0.0, 1.0]) * tube_radius
    angle = 2 * np.pi / n_components
    rotmat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    perim = []
    for _ in range(n_components):
        perim.append(vec)
        vec = np.dot(rotmat, vec)
    poly = Polygon(np.stack(perim, 0))

    # Sweep it along the path
    mesh = trimesh.creation.sweep_polygon(poly, points)
    return add_mesh(
        scene, mesh, name=name, T_mesh_world=pose, color=color, material=material, smooth=smooth
    )


def create_arrow_head(length=0.1, tube_radius=0.005, n_components=30):
    """from https://github.com/BerkeleyAutomation/visualization"""
    radius = tube_radius * 2.0
    height = length * 0.5

    # create a 2D pie out of wedges
    theta = np.linspace(0, np.pi * 2, n_components)
    vertices = np.column_stack((np.sin(theta), np.cos(theta), np.zeros(len(theta)))) * radius

    # the single vertex at the center of the circle
    # we're overwriting the duplicated start/end vertex
    # plus add vertex at tip of cone
    vertices[0] = [0, 0, 0]
    vertices = np.append(vertices, [[0, 0, height]], axis=0)

    # whangle indexes into a triangulation of the pie wedges
    index = np.arange(1, len(vertices)).reshape((-1, 1))
    index[-1] = 1
    faces_2d = np.tile(index, (1, 2)).reshape(-1)[1:-1].reshape((-1, 2))
    faces = np.column_stack((np.zeros(len(faces_2d), dtype=np.int), faces_2d))

    # add triangles connecting to vertex above
    faces = np.append(
        faces,
        np.column_stack(((len(faces_2d) + 1) * np.ones(len(faces_2d), dtype=np.int), faces_2d))[
            :, ::-1
        ],
        axis=0,
    )

    arrow_head = trimesh.Trimesh(faces=faces, vertices=vertices)
    return arrow_head
