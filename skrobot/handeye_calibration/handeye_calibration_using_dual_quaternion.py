from itertools import combinations
from itertools import compress
import math

import numpy as np
from numpy.linalg import norm
from numpy.linalg import svd

from skrobot.dual_quaternion import DualQuaternion
from skrobot.math import normalize_vector
from skrobot.math import outer_product_matrix
from skrobot.math import wxyz2xyzw
from skrobot.math import xyzw2wxyz


def _check_lengths(dq_vec_a, dq_vec_b):
    if len(dq_vec_a) != len(dq_vec_b):
        raise ValueError('length of dq_vec_a and dq_vec_b should be same. '
                         '{} != {}'.format(len(dq_vec_a), len(dq_vec_b)))


def get_aligned_dqs(dq_base_to_ee_vec,
                    dq_camera_to_marker_vec,
                    dq_ee_to_marker_estimated):
    """Return transformations of camera to end-effector

    """
    _check_lengths(dq_base_to_ee_vec, dq_camera_to_marker_vec)
    n = len(dq_base_to_ee_vec)

    dq_marker_to_ee_estimated = dq_ee_to_marker_estimated.inverse
    dq_marker_to_ee_estimated.normalize()
    dq_marker_to_ee_estimated.enforce_positive_q_rot_w()

    dq_camera_to_ee_vec = []
    for i in range(n):
        dq_camera_to_ee = dq_camera_to_marker_vec[i] * \
            dq_marker_to_ee_estimated
        dq_camera_to_ee.normalize()

        if ((dq_camera_to_ee.qr.w < 0.0 and
             dq_base_to_ee_vec[i].qr.w > 0.0) or
            (dq_camera_to_ee.qr.w > 0.0 and
             dq_base_to_ee_vec[i].qr.w < 0.0)):
            dq_camera_to_ee.dq = - dq_camera_to_ee.dq.copy()

        dq_camera_to_ee_vec.append(dq_camera_to_ee)
    dq_camera_to_ee_vec = align_dq_at_index(dq_camera_to_ee_vec)
    return dq_camera_to_ee_vec


def evaluate_alignment(dq_vec_a, dq_vec_b,
                       ransac_orientation_error_threshold_deg=1.0,
                       ransac_position_error_threshold_m=0.02):
    """Calculate translation and rotation error

    """
    _check_lengths(dq_vec_a, dq_vec_b)

    num_dq_vec = len(dq_vec_a)
    inlier_list = [False] * num_dq_vec

    errors_position = np.zeros((num_dq_vec, 1), 'f')
    errors_orientation = np.zeros((num_dq_vec, 1), 'f')
    for i, (pose_a, pose_b) in enumerate(zip(dq_vec_a, dq_vec_b)):
        error_position = pose_a.difference_position(pose_b)
        error_angle_degrees = np.rad2deg(pose_a.difference_rotation(pose_b))
        if (error_angle_degrees < ransac_orientation_error_threshold_deg and
                error_position < ransac_position_error_threshold_m):
            inlier_list[i] = True
        errors_position[i] = error_position
        errors_orientation[i] = error_angle_degrees

    rmse_pos_accumulator = np.sum(np.square(errors_position))
    rmse_orientation_accumulator = np.sum(np.square(errors_orientation))

    rmse_pos = math.sqrt(rmse_pos_accumulator / num_dq_vec)
    rmse_orientation = math.sqrt(rmse_orientation_accumulator / num_dq_vec)

    return (rmse_pos, rmse_orientation, inlier_list)


def make_s_matrix(dq_1, dq_2):
    """Make S matrix

    reference: Hand-Eye Calibration Using Dual Quaternions
    equation (31)

    S = (skew(I(qr1)+I(qr2)) I(qr1)-I(qr2) 0_{3x3}             0_{3x1}      )
        (skew(I(qt1)+I(qt2)) I(qt1)-I(qt2) skew(I(qr1)+I(qr2)) I(qr1)-I(qr2))

    skew -> outer_product_matrix(v):
        return np.array([[0,   -v[2],  v[1]],
                     [v[2],    0,   -v[0]],
                     [-v[1], v[0],     0]])
    """
    scalar_parts_1 = dq_1.scalar
    scalar_parts_2 = dq_2.scalar

    if not np.allclose(scalar_parts_1.dq, scalar_parts_2.dq, atol=5e-2):
        raise ValueError('Scalar parts should always be equal.')

    s_matrix = np.zeros((6, 8), dtype=np.float64)
    s_matrix[0:3, 0:3] = outer_product_matrix(dq_1.qr.xyz + dq_2.qr.xyz)
    s_matrix[0:3, 3] = dq_1.qr.xyz - dq_2.qr.xyz
    s_matrix[0:3, 4:7] = np.zeros((3, 3), dtype=np.float64)
    s_matrix[0:3, 7] = np.zeros(3, dtype=np.float64)
    s_matrix[3:6, 0:3] = outer_product_matrix(dq_1.qd.xyz + dq_2.qd.xyz)
    s_matrix[3:6, 3] = dq_1.qd.xyz - dq_2.qd.xyz
    s_matrix[3:6, 4:7] = outer_product_matrix(dq_1.qr.xyz + dq_2.qr.xyz)
    s_matrix[3:6, 7] = dq_1.qr.xyz - dq_2.qr.xyz
    return s_matrix.copy()


def make_t_matrix(dq_camera_to_marker_vec, dq_base_to_ee_vec):
    """Make T matrix

    reference: Hand-Eye Calibration Using Dual Quaternions
    equation (33)

    T = (S_1.T S_2.T ... S_n.T).T

    """
    n_quaternions = len(dq_camera_to_marker_vec)
    t_matrix = np.zeros((6 * n_quaternions, 8))
    for i in range(n_quaternions):
        t_matrix[i * 6: i * 6 + 6, :] = make_s_matrix(
            dq_camera_to_marker_vec[i], dq_base_to_ee_vec[i])
    return t_matrix.copy()


def align_dq_at_index(dq_vec, align_index=0, enforce_positive_q_rot_w=True):
    """Align DualQuaternion at specified index.

    """
    base_dq = dq_vec[align_index]
    inverse_base_dq = base_dq.inverse.copy()
    n = len(dq_vec)
    aligned_dq_vec = [None] * n
    for i in range(0, n):
        aligned_dq_vec[i] = inverse_base_dq * dq_vec[i].copy()
        if enforce_positive_q_rot_w:
            if aligned_dq_vec[i].qr.w < 0.:
                aligned_dq_vec[i] *= -1.0
    return aligned_dq_vec[align_index:] + aligned_dq_vec[:align_index]


def compute_handeye_calibration_using_dual_quaternion_loop(
        dq_base_to_ee_vec, dq_camera_to_marker_vec,
        scalar_part_tolerance=1e-2,
        enforce_same_non_dual_scalar_sign=True):
    """Compute Handeye Calibration

    reference: Hand-Eye Calibration Using Dual Quaternions

    """
    n_quaternions = len(dq_base_to_ee_vec)

    if not np.allclose(dq_base_to_ee_vec[0].dq,
                       [1, 0, 0, 0, 0, 0, 0, 0], atol=1.0e-8):
        raise ValueError('first index pose should be at origin')
    if not np.allclose(dq_camera_to_marker_vec[0].dq,
                       [1, 0, 0, 0, 0, 0, 0, 0], atol=1.0e-8):
        raise ValueError('first index pose should be at origin')

    if enforce_same_non_dual_scalar_sign:
        for i in range(n_quaternions):
            dq_camera_to_marker = dq_camera_to_marker_vec[i]
            dq_base_to_ee = dq_base_to_ee_vec[i]
            if ((dq_camera_to_marker.qr.w < 0.0 and
                 dq_base_to_ee.qr.w > 0.0) or
                (dq_camera_to_marker.qr.w > 0.0 and
                 dq_base_to_ee.qr.w < 0.0)):
                dq_camera_to_marker_vec[i].dq = \
                    - dq_camera_to_marker_vec[i].dq.copy()

    # checking scalar
    for j in range(n_quaternions):
        dq_base_to_ee = dq_camera_to_marker_vec[j]
        dq_camera_to_marker = dq_base_to_ee_vec[j]

        scalar_parts_base_to_ee = dq_base_to_ee.scalar
        scalar_parts_camera_to_marker = dq_camera_to_marker.scalar

        if not np.allclose(scalar_parts_base_to_ee.dq,
                           scalar_parts_camera_to_marker.dq,
                           atol=scalar_part_tolerance):
            raise ValueError(
                "Mismatch of scalar parts of dual quaternion at index {}: "
                "dq_base_to_ee: {} dq_camera_to_marker: {}".format(
                    j, dq_base_to_ee, dq_camera_to_marker))

    t_matrix = make_t_matrix(dq_base_to_ee_vec, dq_camera_to_marker_vec)
    _, s, V = svd(t_matrix)

    bad_singular_values = False
    for i, singular_value in enumerate(s):
        if i < 6:
            if singular_value < 5e-1:
                bad_singular_values = True
        else:
            if singular_value > 5e-1:
                bad_singular_values = True

    v_7 = V[6, :].copy()
    v_8 = V[7, :].copy()

    u_1 = v_7[0:4].copy()
    u_2 = v_8[0:4].copy()
    v_1 = v_7[4:8].copy()
    v_2 = v_8[4:8].copy()

    a = np.dot(u_1.T, v_1)

    if a == 0.0:
        raise ValueError('Value a should not be zero')

    b = np.dot(u_1.T, v_2) + np.dot(u_2.T, v_1)
    c = np.dot(u_2.T, v_2)
    square_root_term = b * b - 4.0 * a * c

    if square_root_term < - 1.0e-2:
        raise ValueError("square_root_term is too negative: {}"
                         .format(square_root_term))
    if square_root_term < 0.0:
        square_root_term = 0.0

    s_1 = (-b + np.sqrt(square_root_term)) / (2.0 * a)
    s_2 = (-b - np.sqrt(square_root_term)) / (2.0 * a)

    solution_1 = s_1 * s_1 * np.dot(u_1.T, u_1) + 2.0 * \
        s_1 * np.dot(u_1.T, u_2) + np.dot(u_2.T, u_2)
    solution_2 = s_2 * s_2 * np.dot(u_1.T, u_1) + 2.0 * \
        s_2 * np.dot(u_1.T, u_2) + np.dot(u_2.T, u_2)

    if solution_1 > solution_2:
        lambda_2 = np.sqrt(1.0 / solution_1)
        lambda_1 = s_1 * lambda_2
    else:
        lambda_2 = np.sqrt(1.0 / solution_2)
        lambda_1 = s_2 * lambda_2

    tmp = lambda_1 * v_7 + lambda_2 * v_8
    dq_ee_to_marker = DualQuaternion(tmp[0:4], tmp[4:8])
    dq_ee_to_marker = dq_ee_to_marker.normalize()
    if (dq_ee_to_marker.qr.w < 0.0):
        dq_ee_to_marker.dq = - dq_ee_to_marker.dq.copy()
    dq_ee_to_marker = DualQuaternion(xyzw2wxyz(dq_ee_to_marker.dq[0:4]),
                                     xyzw2wxyz(dq_ee_to_marker.dq[4:8]))
    return (dq_ee_to_marker, s, bad_singular_values)


def pose_filter(dq_vec_a, dq_vec_b, dot_product_threshold=0.95):
    """Filtering pose

    """
    _check_lengths(dq_vec_a, dq_vec_b)
    filtered_dq_vec_a = []
    filtered_dq_vec_b = []

    for dq_a, dq_b in zip(dq_vec_a, dq_vec_b):
        screw_axis_a_i, _, _ = dq_a.screw_axis()
        screw_axis_b_i, _, _ = dq_b.screw_axis()
        if not (norm(screw_axis_a_i) <= 1.0e-12 or
                norm(screw_axis_b_i) <= 1.0e-12):
            filtered_dq_vec_a.append(dq_a)
            filtered_dq_vec_b.append(dq_b)

    n = len(filtered_dq_vec_a)
    valid_indices = [True] * n
    for i in range(n):
        dq_a = filtered_dq_vec_a[i]
        dq_b = filtered_dq_vec_b[i]
        screw_axis_a_i, _, _ = dq_a.screw_axis()
        screw_axis_b_i, _, _ = dq_b.screw_axis()
        screw_axis_a_i = normalize_vector(screw_axis_a_i)
        screw_axis_b_i = normalize_vector(screw_axis_b_i)
        for j in range(i + 1, n):
            if valid_indices[j] is False:
                continue
            dq_a_j = filtered_dq_vec_a[j]
            dq_b_j = filtered_dq_vec_b[j]
            screw_axis_a_j, _, _ = dq_a_j.screw_axis()
            screw_axis_b_j, _, _ = dq_b_j.screw_axis()
            screw_axis_a_j = normalize_vector(screw_axis_a_j)
            screw_axis_b_j = normalize_vector(screw_axis_b_j)
            if np.inner(screw_axis_a_i,
                        screw_axis_a_j) > dot_product_threshold or \
                np.inner(screw_axis_b_i,
                         screw_axis_b_j) > dot_product_threshold:
                valid_indices[j] = False
    return list(compress(filtered_dq_vec_a, valid_indices)), \
        list(compress(filtered_dq_vec_b, valid_indices))


def compute_handeye_calibration_using_dual_quaternion(
        dq_base_to_ee_vec,
        dq_camera_to_marker_vec,
        prefilter_dot_product_threshold=0.95,
        ransac_sample_size=1,
        sample_rejection_scalar_part_equality_tolerance=0.01,
        handeye_calibration_scalar_part_equality_tolerance=4e-2,
        min_num_inliers=10):
    """Compute handeye calibration

    reference: Evaluation of Combined Time-Offset Estimation
               and Hand-Eye Calibration on Robotic Datasets

    """
    dq_base_to_ee_vec = align_dq_at_index(dq_base_to_ee_vec)
    dq_camera_to_marker_vec = align_dq_at_index(dq_camera_to_marker_vec)

    # reduce dq pose
    dq_base_to_ee_vec_filtered, dq_camera_to_marker_vec_filtered = pose_filter(
        dq_base_to_ee_vec, dq_camera_to_marker_vec,
        prefilter_dot_product_threshold)
    num_dq_vec_after_filtering = len(dq_camera_to_marker_vec_filtered)

    all_sample_combinations = []
    num_dq_vec = len(dq_base_to_ee_vec)
    indices_set = set(range(num_dq_vec_after_filtering))
    if ransac_sample_size > len(indices_set):
        ransac_sample_size = len(indices_set)

    all_sample_combinations = list(
        combinations(indices_set, ransac_sample_size))
    max_number_samples = len(all_sample_combinations)

    # Result variables:
    best_inlier_index_set = None
    best_num_inliers = 0
    best_rmse_position = np.inf
    best_rmse_orientation = np.inf
    best_estimated_dq_ee_to_marker = None
    best_inlier_flags = None

    sample_number = 0

    while (sample_number < max_number_samples):
        sample_indices = list(all_sample_combinations[sample_number])
        print(sample_indices)
        sample_number += 1

        if ransac_sample_size > 1:
            samples_dq_camera_to_marker = [
                dq_camera_to_marker_vec_filtered[i] for i in sample_indices]
            samples_dq_base_to_ee = [
                dq_base_to_ee_vec_filtered[i] for i in sample_indices]
            aligned_samples_dq_base_to_ee = align_dq_at_index(
                samples_dq_base_to_ee)
            aligned_samples_dq_camera_to_marker = align_dq_at_index(
                samples_dq_camera_to_marker)
            good_sample = True
            for i in range(ransac_sample_size):
                scalar_parts_camera_to_marker = \
                    aligned_samples_dq_camera_to_marker[i].scalar
                scalar_parts_base_to_ee = \
                    aligned_samples_dq_base_to_ee[i].scalar
                if not np.allclose(
                        scalar_parts_camera_to_marker.dq,
                        scalar_parts_base_to_ee.dq,
                        atol=sample_rejection_scalar_part_equality_tolerance):
                    good_sample = False
                    break
            if not good_sample:
                continue

        aligned_dq_base_to_ee = align_dq_at_index(
            dq_base_to_ee_vec, sample_indices[0])
        aligned_dq_camera_to_marker = align_dq_at_index(
            dq_camera_to_marker_vec, sample_indices[0])

        # AX = XB
        # ax = xb
        # a = xbx*
        # a_0 = 1/2 (a + a*) = 1/2 (xbx* + (xbx*)*)
        #     = 1/2 x (b + b*) x* = 1/2 x x* (b + b*)
        #     = b_0
        # if scalar parts are not equal, not inlier
        inlier_flags = [False] * num_dq_vec
        for i in range(num_dq_vec):
            scalar_parts_base_to_ee = aligned_dq_base_to_ee[i].scalar
            scalar_parts_camera_to_marker = aligned_dq_camera_to_marker[i].\
                scalar
            if np.allclose(
                    scalar_parts_camera_to_marker.dq,
                    scalar_parts_base_to_ee.dq,
                    atol=sample_rejection_scalar_part_equality_tolerance):
                inlier_flags[i] = True

        # extract inlier dqs
        inlier_dq_base_to_ee = list(
            compress(aligned_dq_base_to_ee, inlier_flags))
        inlier_dq_camera_to_marker = list(
            compress(aligned_dq_camera_to_marker, inlier_flags))
        num_inliers = len(inlier_dq_base_to_ee)

        if num_inliers < min_num_inliers:
            print("Not enough inliers ({})".format(num_inliers))
            continue

        # calculate handeye calibration
        (dq_ee_to_marker_refined, singular_values, bad_singular_values) = \
            compute_handeye_calibration_using_dual_quaternion_loop(
                inlier_dq_base_to_ee, inlier_dq_camera_to_marker,
                handeye_calibration_scalar_part_equality_tolerance)
        dq_ee_to_marker_refined.normalize()

        dq_vec_camera_to_ee = get_aligned_dqs(
            aligned_dq_base_to_ee,
            aligned_dq_camera_to_marker,
            dq_ee_to_marker_refined)
        dq_vec_base_to_ee = aligned_dq_base_to_ee

        # evaluate handeye calibration result
        (rmse_position_refined,
         rmse_orientation_refined,
         inlier_flags) = evaluate_alignment(
             dq_vec_base_to_ee, dq_vec_camera_to_ee)

        if (rmse_position_refined < best_rmse_position and
                rmse_orientation_refined < best_rmse_orientation):
            best_estimated_dq_ee_to_marker = dq_ee_to_marker_refined
            best_rmse_position = rmse_position_refined
            best_rmse_orientation = rmse_orientation_refined
            best_inlier_index_set = sample_indices
            best_num_inliers = num_inliers
            best_inlier_flags = inlier_flags

            p = dq_ee_to_marker_refined.pose()
            p = np.append(p[:3], wxyz2xyzw(p[3:]))
            print("Found a new best sample: {}\n"
                  "\tNumber of inliers: {}\n"
                  "\tRMSE position:     {:10.4f}\n"
                  "\tRMSE orientation:  {:10.4f}\n"
                  "\tdq_ee_to_marker_refined:    {}\n"
                  "\tpose_ee_to_marker_refined:  {}\n".format(
                      sample_indices, num_inliers, rmse_position_refined,
                      rmse_orientation_refined, dq_ee_to_marker_refined,
                      p))
        else:
            print("Rejected sample: {}\n"
                  "\tNumber of inliers: {}\n"
                  "\tRMSE position:     {:10.4f}\n"
                  "\tRMSE orientation:  {:10.4f}".format(
                      sample_indices, num_inliers, rmse_position_refined,
                      rmse_orientation_refined))

    if best_estimated_dq_ee_to_marker is None:
        raise ValueError('Could not calculate')

    best_estimated_dq_ee_to_marker.enforce_positive_q_rot_w()
    pose_vec = best_estimated_dq_ee_to_marker.pose()
    pose_vec = np.append(pose_vec[:3], wxyz2xyzw(pose_vec[3:]))
    print("Solution found with sample: {}\n"
          "\tNumber of inliers: {}\n"
          "\tRMSE position:     {:10.4f}\n"
          "\tRMSE orientation:  {:10.4f}\n"
          "\tdq_ee_to_marker_refined:    {}\n"
          "\tpose_ee_to_marker_refined:  {}\n"
          "\tTranslation norm:  {:10.4f}".format(
              best_inlier_index_set, best_num_inliers, best_rmse_position,
              best_rmse_orientation, best_estimated_dq_ee_to_marker,
              pose_vec, norm(pose_vec[0:3])))

    indices = list(compress(range(num_dq_vec), best_inlier_flags))
    result = dict(indices=indices,
                  dq=best_estimated_dq_ee_to_marker,
                  pose=pose_vec)
    return result
