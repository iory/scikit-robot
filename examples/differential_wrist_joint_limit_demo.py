#!/usr/bin/env python
"""Differential Wrist Joint Limit Table Visualization Demo.

This demo visualizes the joint limit tables for the DifferentialWristSample robot
in real-time using ViserViewer. It shows:
- 2D plot of the joint limit table (valid region) on the LEFT side of the screen
- Current joint position as a point on the plot
- Whether the current position is within the valid range

The plots update in real-time as you move the robot's joints via sliders or IK.
"""

import base64
import io
import time

import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from skrobot.models import DifferentialWristSample
from skrobot.viewers import ViserViewer


def create_joint_limit_plot_base64(table, current_target_angle, current_dependent_angle,
                                   target_name, dependent_name, figsize=(3.5, 2.8), dpi=100):
    """Create a matplotlib figure and return as base64 PNG string.

    Parameters
    ----------
    table : JointLimitTable
        The joint limit table to visualize.
    current_target_angle : float
        Current angle of the target joint (radians).
    current_dependent_angle : float
        Current angle of the dependent joint (radians).
    target_name : str
        Name of the target joint.
    dependent_name : str
        Name of the dependent joint.
    figsize : tuple
        Figure size in inches.
    dpi : int
        DPI for the figure.

    Returns
    -------
    str
        Base64 encoded PNG image string.
    bool
        Whether the current position is within limits.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot valid region
    x = np.rad2deg(table.sample_angles)
    y_min = np.rad2deg(table.min_angles)
    y_max = np.rad2deg(table.max_angles)

    ax.fill_between(x, y_min, y_max, alpha=0.3, color='#4A90D9', label='Valid region')
    ax.plot(x, y_min, color='#2E5C8A', linewidth=1.5)
    ax.plot(x, y_max, color='#2E5C8A', linewidth=1.5)

    # Current position
    current_x = np.rad2deg(current_target_angle)
    current_y = np.rad2deg(current_dependent_angle)

    # Check if within limits
    min_limit = table.min_angle_function(current_target_angle)
    max_limit = table.max_angle_function(current_target_angle)
    is_valid = min_limit <= current_dependent_angle <= max_limit

    # Plot current position
    color = '#28A745' if is_valid else '#DC3545'
    marker = 'o' if is_valid else 'x'
    markersize = 80 if is_valid else 100
    if is_valid:
        ax.scatter([current_x], [current_y], c=color, s=markersize, marker=marker,
                   zorder=5, edgecolors='white', linewidths=1.5)
    else:
        # 'x' marker doesn't support edgecolors
        ax.scatter([current_x], [current_y], c=color, s=markersize, marker=marker,
                   zorder=5, linewidths=2)

    # Draw vertical line at current target angle
    ax.axvline(x=current_x, color='#888888', linestyle='--', alpha=0.5, linewidth=1)

    # Draw horizontal lines for current limits
    current_min_deg = np.rad2deg(min_limit)
    current_max_deg = np.rad2deg(max_limit)
    ax.axhline(y=current_min_deg, color='#FFA500', linestyle='-', alpha=0.8, linewidth=1.5)
    ax.axhline(y=current_max_deg, color='#FFA500', linestyle='-', alpha=0.8, linewidth=1.5)

    ax.set_xlabel(f'{target_name} [deg]', fontsize=9)
    ax.set_ylabel(f'{dependent_name} [deg]', fontsize=9)
    ax.set_title(f'{target_name} → {dependent_name}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # Add status indicator
    status_text = '● OK' if is_valid else '✗ OUT'
    status_color = '#28A745' if is_valid else '#DC3545'
    ax.text(0.98, 0.98, status_text, transform=ax.transAxes, fontsize=10,
            fontweight='bold', color=status_color,
            verticalalignment='top', horizontalalignment='right')

    # Add current value text
    value_text = f'{dependent_name}: {current_y:.1f}°'
    ax.text(0.02, 0.02, value_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05,
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64, is_valid


def create_overlay_html(img1_base64, img2_base64, is_valid1, is_valid2):
    """Create HTML for left-side overlay with two plots.

    Parameters
    ----------
    img1_base64 : str
        Base64 encoded first plot image.
    img2_base64 : str
        Base64 encoded second plot image.
    is_valid1 : bool
        Whether first joint is within limits.
    is_valid2 : bool
        Whether second joint is within limits.

    Returns
    -------
    str
        HTML content string.
    """
    border1 = '#28A745' if is_valid1 else '#DC3545'
    border2 = '#28A745' if is_valid2 else '#DC3545'

    html = f'''
<style>
.jlt-overlay {{
    position: fixed !important;
    top: 10px !important;
    left: 10px !important;
    z-index: 99999 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    pointer-events: none !important;
}}
.jlt-plot {{
    background: white !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15) !important;
    overflow: hidden !important;
    border: 3px solid;
}}
.jlt-plot img {{
    display: block !important;
    max-width: 320px !important;
}}
</style>
<div class="jlt-overlay">
    <div class="jlt-plot" style="border-color: {border1};">
        <img src="data:image/png;base64,{img1_base64}"/>
    </div>
    <div class="jlt-plot" style="border-color: {border2};">
        <img src="data:image/png;base64,{img2_base64}"/>
    </div>
</div>
'''
    return html


def main():
    print("=" * 60)
    print("Differential Wrist Joint Limit Table Visualization")
    print("=" * 60)

    # Load robot
    print("\nLoading DifferentialWristSample robot...")
    robot = DifferentialWristSample(use_joint_limit_table=True)
    robot.reset_manip_pose()

    # Get wrist joints and their limit tables
    wrist_y = robot.WRIST_JOINT_Y
    wrist_r = robot.WRIST_JOINT_R

    # Get joint limit tables
    table_y = wrist_y.joint_min_max_table  # Y's limits depend on R
    table_r = wrist_r.joint_min_max_table  # R's limits depend on Y

    print("\nJoint limit tables:")
    print("  WRIST_JOINT_Y: limits depend on WRIST_JOINT_R")
    print("  WRIST_JOINT_R: limits depend on WRIST_JOINT_Y")

    # Create viewer
    print("\nStarting ViserViewer...")
    viewer = ViserViewer(enable_ik=True)
    viewer.add(robot)

    # Get viser server
    server = viewer._server

    # Create initial plots
    y_angle = wrist_y.joint_angle()
    r_angle = wrist_r.joint_angle()

    img1_base64, is_valid1 = create_joint_limit_plot_base64(
        table_y, r_angle, y_angle, "WRIST_R", "WRIST_Y")
    img2_base64, is_valid2 = create_joint_limit_plot_base64(
        table_r, y_angle, r_angle, "WRIST_Y", "WRIST_R")

    # Add HTML overlay on the left side
    html_content = create_overlay_html(img1_base64, img2_base64, is_valid1, is_valid2)
    html_handle = server.gui.add_html(html_content)

    # Show viewer
    viewer.show()

    print("\n" + "=" * 60)
    print("Visualization started!")
    print("=" * 60)
    print("\nOpen your browser at: http://localhost:8085")
    print("\nFeatures:")
    print("  - LEFT side: Joint limit table plots (green=OK, red=OUT)")
    print("  - Green dot: Current position within limits")
    print("  - Red X: Current position outside limits")
    print("  - Orange lines: Current dynamic limits")
    print("\nInstructions:")
    print("  1. Move WRIST_JOINT_Y slider → see WRIST_JOINT_R limits change")
    print("  2. Move WRIST_JOINT_R slider → see WRIST_JOINT_Y limits change")
    print("  3. Drag the end-effector gizmo to test IK with constraints")
    print("\nPress Ctrl+C to exit.")

    # Update loop
    last_y_angle = None
    last_r_angle = None

    try:
        while True:
            # Get current joint angles
            y_angle = wrist_y.joint_angle()
            r_angle = wrist_r.joint_angle()

            # Only update plots if angles changed significantly
            if (last_y_angle is None or last_r_angle is None or
                abs(y_angle - last_y_angle) > 0.005 or
                abs(r_angle - last_r_angle) > 0.005):

                # Create updated plots
                img1_base64, is_valid1 = create_joint_limit_plot_base64(
                    table_y, r_angle, y_angle, "WRIST_R", "WRIST_Y")
                img2_base64, is_valid2 = create_joint_limit_plot_base64(
                    table_r, y_angle, r_angle, "WRIST_Y", "WRIST_R")

                # Update HTML overlay
                html_content = create_overlay_html(
                    img1_base64, img2_base64, is_valid1, is_valid2)
                html_handle.content = html_content

                last_y_angle = y_angle
                last_r_angle = r_angle

            time.sleep(0.05)  # 20 Hz update rate

    except KeyboardInterrupt:
        print("\n\nExiting...")
        viewer.close()


if __name__ == "__main__":
    main()
