import argparse

import numpy as np
import trimesh
import trimesh.viewer

import skrobot


def show_with_rotation(scene, step=None, init_angles=None, **kwargs):
    if step is None:
        step = (0, np.deg2rad(1), 0)
    if init_angles is None:
        init_angles = (0, 0, 0)

    step = np.asarray(step, dtype=float)
    init_angles = np.asarray(init_angles, dtype=float)

    def callback(scene):
        if hasattr(scene, 'angles'):
            scene.angles += step
        else:
            scene.angles = init_angles
        scene.set_camera(angles=scene.angles)

    return trimesh.viewer.SceneViewer(scene=scene, callback=callback, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--robot',
        choices=['kuka', 'fetch', 'pr2', 'panda'],
        default='kuka',
        help='robot',
    )
    args = parser.parse_args()

    if args.robot == 'kuka':
        robot = skrobot.models.Kuka()
    elif args.robot == 'fetch':
        robot = skrobot.models.Fetch()
    elif args.robot == 'pr2':
        robot = skrobot.models.PR2()
    elif args.robot == 'panda':
        robot = skrobot.models.Panda()
    else:
        raise ValueError('unsupported robot')

    print('link_list:')
    for link in robot.link_list:
        print('  {}'.format(link))
    print('joint_list:')
    for joint in robot.joint_list:
        print('  {}'.format(joint))

    scene = trimesh.Scene()
    geom = trimesh.creation.box((2, 2, 0.01))
    geom.visual.face_colors = (0.75, 0.75, 0.75)
    scene.add_geometry(geom)

    for link in robot.link_list:
        transform = link.worldcoords().T()
        scene.add_geometry(link.visual_mesh, transform=transform)

    show_with_rotation(
        scene,
        init_angles=(np.deg2rad(45), np.deg2rad(0), 0),
        step=(0, 0, np.deg2rad(1)),
        resolution=(800, 800),
    )


if __name__ == '__main__':
    main()
