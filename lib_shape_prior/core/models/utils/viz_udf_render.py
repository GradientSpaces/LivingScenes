# render the results of the UDF, the input and the results
import sys, os

sys.path.append(os.path.dirname(__file__))

import imageio
from pyrender_helper_v2 import render
import numpy as np


def viz_input_and_recon(
    input=None,
    output=None,
    views=2,
    pitch=np.pi / 4,
    yaw_base=np.pi / 3,
    cam_dist=1.5,
    points_radius=0.02,
    input_color=[1.0, 0.5, 0.5],
    output_color=[0.5, 1.0, 0.5],
    shape=(300, 300),
    pcl_max_num=8000,
):
    assert input is not None or output is not None
    if input is not None:
        if len(input) > pcl_max_num:
            viz_input = input[np.random.choice(len(input), pcl_max_num)]
        else:
            viz_input = input
    if output is not None:
        if len(output) > pcl_max_num:
            viz_output = output[np.random.choice(len(output), pcl_max_num)]
        else:
            viz_output = output

    ret = []
    for vid in range(views):
        yaw = yaw_base + 2 * np.pi / views * vid
        rgb = []
        if input is not None:
            rgb.append(
                render(
                    pcl_list=[viz_input],
                    pcl_radius_list=[points_radius],
                    pcl_color_list=[input_color],
                    cam_dist=cam_dist,
                    cam_angle_pitch=pitch,
                    cam_angle_yaw=yaw,
                    shape=shape,
                )
            )
        if output is not None:
            rgb.append(
                render(
                    pcl_list=[viz_output],
                    pcl_radius_list=[points_radius],
                    pcl_color_list=[output_color],
                    cam_dist=cam_dist,
                    cam_angle_pitch=pitch,
                    cam_angle_yaw=yaw,
                    shape=shape,
                )
            )
        # joint_rgb = render(
        #     pcl_list=[output, input],
        #     pcl_radius_list=[0.01, 0.01],
        #     pcl_color_list=[output_color + [0.3], input_color + [1.0]],
        #     cam_dist=cam_dist,
        #     cam_angle_pitch=pitch,
        #     cam_angle_yaw=yaw,
        #     shape=shape,
        # )
        # rgb = np.concatenate([input_rgb, output_rgb, joint_rgb], 1)
        rgb = np.concatenate(rgb, 1)
        mar = 3
        rgb[:mar, :], rgb[-mar:, :] = 0.0, 0.0
        rgb[:, :mar], rgb[:, -mar:] = 0.0, 0.0
        ret.append(rgb)
        # imageio.imsave("./debug/input.png", input_rgb)
        # imageio.imsave("./debug/output.png", output_rgb)
        # imageio.imsave("./debug/joint.png", joint_rgb)
    ret = np.concatenate(ret, 0)
    return ret


if __name__ == "__main__":
    rgb = viz_input_and_recon(input=np.random.rand(10, 3), output=np.random.rand(10, 3))
    print(rgb.shape)
