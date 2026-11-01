#!/usr/bin/env python
"""Drive Fetch in Isaac Sim through the same calls as a real robot.

``robot`` is the model you plan with; ``ri`` sends its posture to the
simulated robot and reads back what the robot did. With ``--viewer`` the
simulation is shown at http://localhost:8765/, rendered by Isaac Sim's RTX
and orbited with the mouse. In Colab, show that page in the notebook with
``google.colab.output.serve_kernel_port_as_iframe(8765)``.

Needs Isaac Sim (Python 3.12) and, for the viewer, websockets:

    pip install "isaacsim[all,extscache]" websockets \
        --extra-index-url https://pypi.nvidia.com
"""

import argparse

import skrobot
from skrobot.interfaces import IsaacSimRobotInterface


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--viewer', action='store_true',
                        help='serve the RTX view in a browser')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--scene', default=None,
                        help='environment, e.g. full_warehouse (see '
                        'skrobot.interfaces._isaacsim.SCENES)')
    parser.add_argument('--loops', type=int, default=1,
                        help='times to repeat the motion (0: forever)')
    args = parser.parse_args()

    robot = skrobot.models.Fetch()
    ri = IsaacSimRobotInterface(
        robot, planar_base=True, scene=args.scene,
        viewer=args.viewer, viewer_port=args.port,
        cameras={'head': {'link': 'head_camera_rgb_optical_frame',
                          'fovy': 50, 'width': 640, 'height': 480}})
    if args.viewer:
        print('viewer: http://localhost:{}/'.format(args.port))

    robot.reset_pose()
    home = robot.angle_vector().copy()
    robot.shoulder_pan_joint.joint_angle(-0.8)
    robot.elbow_flex_joint.joint_angle(-1.2)
    robot.torso_lift_joint.joint_angle(0.3)
    reach = robot.angle_vector().copy()

    i = 0
    while args.loops == 0 or i < args.loops:
        ri.angle_vector(home, 3.0)
        ri.wait_interpolation()
        ri.angle_vector(reach, 3.0)
        ri.wait_interpolation()
        ri.go_pos(0.3, 0.0, 0.5)
        ri.go_pos(-0.3, 0.0, -0.5)
        i += 1

    robot.angle_vector(ri.angle_vector())
    print('simulated {:.1f} s, base at {}'.format(ri.time,
                                                  ri.odom.worldpos()))
    rgb = ri.get_image('head')
    print('head camera: {} image, mean {:.0f}'.format(rgb.shape, rgb.mean()))
    if args.viewer:
        print('holding the last pose for the viewer; Ctrl-C to quit')
        ri.spin()


if __name__ == '__main__':
    main()
