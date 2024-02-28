import numpy as np
import trimesh
import pyrender

from scipy.spatial.transform import Rotation

scale = 1
image_height = 100 * scale
image_width = 100 * scale
fx = 100 * scale
fy = 100 * scale
cx = 50 * scale
cy = 50 * scale
k = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float32).reshape((3, 3))

# the pose for render front-facing images of mesh
camera_pose = np.array([[-0.97456526,  0.04894683, -0.21869332, -0.32803997],
       [-0.00465062,  0.97122945,  0.23810025,  0.35715036],
       [ 0.22405565,  0.23306129, -0.94629884, -1.41944827],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

def gen_random_poses(n_views):
    cam_poses = []
    
    for _ in range(n_views):
        Rot_mat = Rotation.random().as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = Rot_mat
        xyz = Rot_mat @ np.linalg.inv(k) @ np.array([[cx],[cy],[1]])
        pose[:3, [3]] = xyz * 1.5
        cam_poses.append(pose)
    return cam_poses

def render_mesh(mesh, out_path):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    sl = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=5.0,
                        innerConeAngle=0.05, outerConeAngle=1.5)
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    scene.add(mesh)
    scene.add(camera, pose=camera_pose)
    scene.add(sl, pose=camera_pose)
    r = pyrender.OffscreenRenderer(image_height, image_width)
    rgb, _ = r.render(scene) 
    from PIL import Image
    im = Image.fromarray(rgb)
    im.save(out_path)


def render_depth(tri_mesh, n_views):
    
    cam_poses = gen_random_poses(n_views)
    
    # camera = pyrender.PerspectiveCamera(yfov=2 * np.arctan(fy/(image_height/2)), aspectRatio=image_height/image_width)
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    
    depth_maps = []
    pcls = []
    T_cw_list = []
    
    for pose in cam_poses:
        scene.clear()
        # camera = pyrender.PerspectiveCamera(yfov=2 * np.arctan(1/2.), aspectRatio=1.0)
        camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        
        scene.add(mesh)
        scene.add(camera, pose=pose)
        r = pyrender.OffscreenRenderer(image_height, image_width)

        rgb, depth = r.render(scene)
        r.delete()
        depth_maps.append(depth)

        pcl_cam = pointcloud(depth)
        
        # https://github.com/colmap/colmap/issues/704#issuecomment-954161261 pyrender pose different from colmap
        R = pose[:3, :3]
        T = pose[:3, [3]]
        T[:,1:3] *= -1
        pcl_cam[:,1:3] *= -1
        pcl = R @ (pcl_cam.T ) + T
        pcl = pcl.T
        pcls.append(pcl)
        
        T_cw = np.eye(4)
        T_cw[:3, :3] = R
        T_cw[:3, [3]] = T
        T_cw_list.append(T_cw)

    return pcls, T_cw_list

def pointcloud(depth):
        # fy = fx = 0.5 / np.tan(fov * 0.5) # assume aspectRatio is one.
        height = depth.shape[0]
        width = depth.shape[1]

        mask = np.where(depth > 0)
        d = depth[mask]
        x = mask[1] * d
        y = mask[0] * d
        
        uvd = np.concatenate([x[:, None], y[:, None], d[:, None]], axis=1)
        # xyz_c = (np.linalg.inv(k) @ uv.T).T
        # xyz_c = np.concatenate([xyz_c, np.ones_like(d[:, None])], axis=1)
        xyz_c = (np.linalg.inv(k) @ uvd.T).T
        
        return xyz_c

if __name__ == "__main__":
    tri_mesh = trimesh.load('/scratch/liyzhu/MA_Thesis/EFEM/data/room_4cate_v3_watertight/03001627/1d0abb4d48b46e3f492d9da2668ec34c.obj')
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    sl = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=5.0,
                        innerConeAngle=0.05, outerConeAngle=1.5)
    # sl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    # camera_pose = gen_random_poses(500)
    # camera_pose = np.load('/scratch/liyzhu/MA_Thesis/EFEM/utils/camera_poses.npy')
    scene.add(mesh)
    scene.add(camera, pose=camera_pose)
    scene.add(sl, pose=camera_pose)
    r = pyrender.OffscreenRenderer(image_height, image_width)
    rgb, _ = r.render(scene) 
    from PIL import Image
    im = Image.fromarray(rgb)
    im.save(f'/scratch/liyzhu/MA_Thesis/EFEM/utils/images/render.png')





