import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.spatial.transform import Rotation as R
import sci3d as s3d


window_width = 256
real_E = None
real_dr = None
real_dt = None
unprojected_points_all_noround = None
normalization_transforms_all_noround = None


def load_obj(path):
    with open(path, 'r') as f:
        obj = f.readlines()

    vertices = np.array([
        [float(v) for v in line.split(' ')[1:]] for line in obj if line[0] == 'v'
     ])
    indices = np.array([
        [int(v) - 1 for v in line.split(' ')[1:]] for line in obj if line[0] == 'f'
    ])

    return vertices, indices


def homognorm(x):
    return x / x[-1]


def mean_std(val):
    return np.mean(val), np.std(val)


def min_max(val):
    return np.min(val), np.max(val)


def skew_symmetric_of(v):
    v = np.squeeze(v)
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def normalize(v):
    return v / np.linalg.norm(v)


def transform_points(points, rotation, translation):
    points = (rotation @ points.T + translation).T
    return points


def relative_transform(rotation_to,
                       translation_to,
                       rotation_from,
                       translation_from):
    delta_rot = rotation_to @ rotation_from.T
    delta_t = translation_to - delta_rot @ translation_from
    return delta_rot, delta_t


def project_points(points, camera, viewport, do_round=True):
    # Homogeneous normalization
    points = (camera @ points.T).T
    points = points / points[:, 2:3]
    points = (viewport @ points.T).T

    if do_round:
        points = np.round(points)

    return points


def get_normalization_transform(points):
    """
    Find N such that points @ N.T has zero mean and mean distance to origin of 2
    :param points:
    :return:
    """
    assert(points.shape[1] == 3)
    translate = -np.mean(points[:, :2], 0, keepdims=True)
    assert(translate.shape == (1, 2))
    centered_pts = points[:, :2] + translate
    # |a_i|^2 = a_i dot a_i
    rms = np.sqrt(np.mean(np.einsum('ij,ij->i', centered_pts, centered_pts)))
    scale_value = np.sqrt(2) / rms
    scale = np.eye(2) * scale_value
    # [scale | translate]
    transform = np.concatenate([scale, scale_value * translate.T], axis=1)
    assert(transform.shape == (2, 3))

    # Append [0, 0, 1]
    transform = np.concatenate([transform, [[0, 0, 1]]], axis=0)
    assert(transform.shape == (3, 3))

    return transform


def unproject_points(points, camera, viewport):
    points = (np.linalg.inv(camera) @ np.linalg.inv(viewport) @ points.T).T
    return points


def random_transform():
    random_quat = np.random.normal(size=(4,))
    rotation = R.from_quat(random_quat).as_matrix()
    translation = np.random.normal(size=(3, 1)) * 0.125/2 + np.array([[0, 0, -10.25]]).T

    return rotation, translation


def plot2d(points,
           window_width,
           window_height,
           recenter=True):
    fig, ax = plt.subplots()
    plt.scatter(points[:, 0], points[:, 1])
    ax.set_aspect(1)
    if recenter:
        ax.set_xlim([0, window_width-1])
        ax.set_ylim([0, window_height-1])
    plt.gca().invert_yaxis()
    plt.show()


def plot3d(points, indices):
    s3d.figure()
    s3d.mesh(points.astype(np.float32), indices.astype(np.uint32))
    while s3d.get_window_count() > 0:
        time.sleep(0.1)


def get_essential(pts_a, pts_b):
    """
    We start with the fundamental matrix. For that, pts_a and pts_b are assumed to be
    normalized:
    pts_a = Na*V*h(P*A)
    pts_b = Nb*V*h(P*M*A), with B=M*A

    If we had the perfect projected coordinates a' and b', then
       a'.T * E * b' = 0.
    What we have are the normalized coordinates a and b. With some
    simple manipulation of the equation above, we can replace a' and b'
    by a and b, respectively:
       a'.T * Na.T * Na.T^-1 * E Nb^-1 * Nb * b' = 0,
       a * Na.T^-1 * E * Na^-1 * b = 0,
       a * P * b = 0.
    So the P found by this procedure should be convertible to the
    matrix E by:
       E = Na.T @ P @ Nb

    :param pts_a:
    :param pts_b:
    :return:
    """
    assert(pts_a.shape[0] == pts_b.shape[0])  # n_points
    assert(pts_a.shape[1] == pts_b.shape[1] == 3)

    normalization_a = get_normalization_transform(pts_a)
    normalization_b = get_normalization_transform(pts_b)

    pts_a_n = (pts_a @ normalization_a.T)
    pts_b_n = (pts_b @ normalization_b.T)

    # a @ b.T
    a_times_b = np.array([
        (pt_a_n[:, np.newaxis] @ pt_b_n[np.newaxis, :]).flatten()
        for pt_a_n, pt_b_n in zip(pts_a_n, pts_b_n)
    ])

    u, s, vh = scipy.linalg.svd(a_times_b)
    E_raw = vh[-1, :].reshape((3, 3))

    # Make E have rank 2
    ue, se, vhe = scipy.linalg.svd(E_raw)
    E = ue @ np.diag([se[0], se[1], 0]) @ vhe

    # Show that a.T @ E @ b = 0 (ideally; may not occur in real life)
    ic('a.T @ estimated E @ b', mean_std(np.abs([
        pt_a[np.newaxis, :] @ E @ pt_b[:, np.newaxis]
        for pt_a, pt_b in zip(pts_a, pts_b)
    ])))

    # Show that a.T @ real_E @ b = 0
    ic('a.T @ real E @ b', mean_std(np.abs([
        pt_a[np.newaxis, :] @ real_E @ pt_b[:, np.newaxis]
        for pt_a, pt_b in zip(pts_a, pts_b)
    ])))

    # Scatter plot point clouds
    pts_a = pts_a_n
    pts_b = pts_b_n
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(pts_a[:, 0], pts_a[:, 1], s=0.1)
    ax[1].scatter(pts_b[:, 0], pts_b[:, 1], s=0.1)

    # Functions to compute line coefficients
    line_eq = lambda i: pts_a[i] @ E
    get_y = lambda x, i: -(line_eq(i)[2]+line_eq(i)[0]*x)/line_eq(i)[1]
    get_line_pts = lambda i: ((-1, get_y(-1, i)), (1, get_y(1, i)))

    # Draw epipolar lines
    num_lines = 10
    colormap = plt.cm.get_cmap('jet', num_lines+1)
    points_rnd_idx = np.random.randint(0, pts_a.shape[0], size=num_lines)
    for i in range(num_lines):
        color = colormap(i)
        idx = points_rnd_idx[i]
        # Draw A point
        ax[0].scatter(pts_a[idx:idx+1, 0], pts_a[idx:idx+1, 1], color=color, s=10.0, marker='o')
        # Draw epipolar line corresponding to A on right figure
        ax[1].axline(*get_line_pts(idx), color=color, linewidth=0.5)

    # Plot intersection of some lines
    ax[1].scatter(*homognorm(np.cross(line_eq(points_rnd_idx[0]), line_eq(points_rnd_idx[1])))[:2], s=50.0, marker='o')
    # The intersection of all lines should be the projection of the origin of the other camera
    origin_a_in_b = normalization_transforms_all_noround[1] @ homognorm(-real_dr.T @ real_dt)
    ax[1].scatter(*origin_a_in_b.flatten()[:2], s=50, marker='*', color=(0,0,0))

    # Adjust plot style
    for axi in ax:
        axi.set_aspect(1)
        axi.set_xlim([-20, 20])
        axi.set_ylim([-20, 20])
        axi.invert_yaxis()

    plt.show()
    exit()

    return E


def retrieve_rt(E, gt_r, gt_t):
    u, s, vh = scipy.linalg.svd(E)
    ic(gt_r, normalize(gt_t))
    ic(u, s, vh)
    """
    s[0]=s[1].
    Proof:
    E=[t]xR. Thus, for any unit vector v perpendicular to t,
        v.T @ E = v.T @ [t]x @ R.
    In particular, |v.T @ [t]x| = |[-t]x @ v| = |v| * |-t| * sin(90) = |t|.
    Whatever v' results from v.T @ [t]x, |v' @ R| = |v'| = |t| as R doesn't change the vector's norm.
    Therefore, the singular values associated with the vectors that create the plane perpendicular
    to t must be the same, s[0]=s[1].
    """
    t_direction = -u[:, 2:3]

    cross_prods = [
        skew_symmetric_of(E[:, i] / s[0])
        for i in range(3)
    ]
    #ic(gt_r[:, 0:1] - t_direction * (np.dot(gt_r[:, 0:1].T, t_direction)))
    #ic(cross_prods[0] @ t_direction)
    #ic(t_direction.T @ cross_prods[0].T @ cross_prods[0] @ t_direction + gt_r[:, 0:1].T @ t_direction @ t_direction.T @ t_direction @ t_direction.T @ gt_r[:, 0:1])
    M = t_direction @ t_direction.T @ t_direction @ t_direction.T

    i_of = lambda i: np.squeeze(1 - t_direction.T @ cross_prods[i].T @ cross_prods[i] @ t_direction)
    o_of = lambda i, j: np.squeeze(- t_direction.T @ cross_prods[i].T @ cross_prods[j] @ t_direction)

    ic(gt_r, gt_r @ normalize(gt_t), scipy.linalg.svd(M))
    ic(np.sqrt(1 - np.linalg.norm(E / s[0], axis=0)**2))
    io_mat = np.array([
        [i_of(0), o_of(0, 1), o_of(0, 2)],
        [o_of(1, 0), i_of(1), o_of(1, 2)],
        [o_of(2, 0), o_of(2, 1), i_of(2)],
    ])
    #ic(io_mat)

    rhs = np.concatenate([cross_prod @ t_direction for cross_prod in cross_prods], axis=1)
    lhs = np.eye(3) - t_direction @ t_direction.T
    ic(lhs @ gt_r, rhs)
    """
    """


def sfm():
    """
    Conventions:

    - post homogeneous normalization:
      * -1,1 on x; y range depend on image aspect ratio (width/height).
      * x: point right
      * y: point up
    - viewport:
      * y pointing down

    :return:
    """
    np.random.seed(12342)

    points, indices = load_obj('/home/claudio/Downloads/FLAME2020/flame.obj')
    points[:, 0] = np.maximum(points[:, 0], 0.5 * points[:, 0])
    viewport = np.array([
        [0.5*window_width, 0, 0.5*window_width],
        [0, -0.5*window_width, 0.5*window_width],
        [0, 0, 1]
    ])
    camera = np.array([
        [-5, 0, 0],
        [0, -5, 0],
        [0, 0, 1],
    ])

    camera_poses = [
        random_transform()
        for _ in range(2)
    ]
    transformed_points_all = [
        transform_points(points, rotation, translation)
        for rotation, translation in camera_poses
    ]
    dr, dt = relative_transform(*camera_poses[0], *camera_poses[1])
    #plot3d(transformed_points_all[1], indices)
    #plot3d(transform_points(transformed_points_all[1], dr, dt), indices)
    #plot3d(transformed_points_all[0], indices)

    # Rounded (real world) version
    projected_points_all = [
        project_points(transformed_points, camera, viewport)
        for transformed_points in transformed_points_all
    ]
    unprojected_points_all = [
        unproject_points(projected_points, camera, viewport)
        for projected_points in projected_points_all
    ]

    # No round version
    global unprojected_points_all_noround
    projected_points_all_noround = [
        project_points(transformed_points, camera, viewport, do_round=False)
        for transformed_points in transformed_points_all
    ]
    unprojected_points_all_noround = [
        unproject_points(projected_points, camera, viewport)
        for projected_points in projected_points_all_noround
    ]
    global normalization_transforms_all_noround
    normalization_transforms_all_noround = [
        get_normalization_transform(unprojected_points)
        for unprojected_points in unprojected_points_all_noround
    ]

    global real_E
    global real_dr
    global real_dt
    real_E = skew_symmetric_of(dt) @ dr
    real_dr = dr
    real_dt = dt
    """
    By exploring E.T@E, we find that v(t.t)-E.t@E@v = r.t @ t(t . r@v) for any vector v
    This means we can find r.t @ t*k (i.e. scaled t transformed by r.t)
    """
    #rv = normalize(np.random.normal(size=(3,1)))
    #t2=dt.T @ dt
    #ic(dr.T@dt@dt.T @ dr @ rv, (rv*t2-real_E.T@real_E@rv))

    # What does [tx]@v look like?
    #plot3d((skew_symmetric_of(dt)@points.T).T, indices)
    # Whats the SVD of skew symmetric only?
    #u,s,v = scipy.linalg.svd(skew_symmetric_of(dt))
    """
    Retrieving rotation r:
    We know E = [tx]@r. We also know that the cross product matrix [tx]
    represents Rt.T @ R90 @ S @ Rt, where the rotation Rt aligns the axis t with Z,
    S scales by |t| on X,Y, and by 0 on Z (so it squeezes the space), and R90 is a
    90deg rotation around Z.
    
    By plugging everything together, E=Rt.T @ R90 @ S @ Rt @ r.
    With an SVD decomposition, we find E=U@S@V. However, that doesn't mean SVD will find
    U=Rt.T @ R90 and V=Rt @ r, since the two vectors that create the base for the 2D squeezed
    space created by E have the same intensity. In other words, the orthonormal vectors
    perpendicular to t can have any 2D rotation around t (which is a rotation K around Z
    when Rt is applied). This means:
      E = U @ S @ V
        = Rt.T @ R90 @ S @ K.T @ K @ Rt @ r
        = Rt.T @ R90 @ K.T @ S @ K @ Rt @ r
    Note that S @ K.T = K.T @ S due to K being a 2d rotation around Z and S being a
    scaling (diagonal) matrix.
    
    Thus, U=Rt.T @ R90 @ K.T=Rt.T @ K.T @ R90 (because 2d rotations are commutative) and
    V=K @ Rt @ r. We can finally retrieve r:
      r = U @ R90.T @ V
        = Rt.T @ K.T @ R90 @ R90.T @ K @ Rt @ r
        = Rt.T @ K.T @ K @ Rt @ r
        = Rt.T @ Rt @ r
        = r
    """
    #u2,s2,v2 = scipy.linalg.svd(real_E)
    #ic(u,s,v, u2,s2,v2)
    #ic(u, u2)
    #ic(v@dr, v2)
    #ic(u.T@u2 @ v@dr@v2.T)
    #ic(u@np.diag(s)@v@dr, real_E)

    # What are the statistical properties of the error?
    #err = unprojected_points_all[0] - unprojected_points_all_noround[0]
    #ic(err, np.mean(err, 0), np.std(err, 0))
    #plt.hist(err[:, 0], bins=40)
    #plt.hist(err[:, 1], bins=40)
    #plt.show()

    """
    fig, ax = plt.subplots(3, 3)
    gt = np.array([
        (a_[:, np.newaxis] @ b_[np.newaxis, :]).flatten()
        for a_, b_ in zip(unprojected_points_all_noround[0],
                          unprojected_points_all_noround[1])
    ])
    eet = np.array([
        (a_[:, np.newaxis] @ b_[np.newaxis, :]).flatten()
        for a_, b_ in zip(unprojected_points_all[0],
                          unprojected_points_all[1])
    ]) - gt
    for i in range(3):
        for j in range(3):
            ax[i, j].hist(eet[:, i*3+j], bins=40)
    plt.show()
    """

    # What does (b+e)*(a+e).T - b*a.T look like?
    #err1=np.random.uniform(-1, 1, (unprojected_points_all[0].shape[0], 3))
    #err2=np.random.uniform(-1, 1, (unprojected_points_all[0].shape[0], 3))
    #a=np.random.uniform(0, 256, (unprojected_points_all[0].shape[0], 3))
    #b=np.random.uniform(0, 256, (unprojected_points_all[0].shape[0], 3))
    #gt = np.array([
    #    (a_[:, np.newaxis] @ b_[np.newaxis, :]).flatten()
    #    for a_, b_, e1, e2 in zip(a, b, err1, err2)
    #])
    #eet = np.array([
    #    ((a_[:, np.newaxis] + e1[:, np.newaxis]) @ (b_[np.newaxis, :] + e2[np.newaxis, :]) -
    #      a_[:, np.newaxis] @ b_[np.newaxis, :]).flatten()
    #    for a_, b_, e1, e2 in zip(a, b, err1, err2)
    #])
    #ic(gt, eet, np.mean(eet / gt, 0))
    #fig, ax = plt.subplots(3, 3)
    #for i in range(3):
    #    for j in range(3):
    #        ax[i, j].hist(eet[:, i*3+j], bins=40)
    #plt.show()

    #E = get_essential(unprojected_points_all_noround[0], unprojected_points_all_noround[1])
    E = get_essential(unprojected_points_all[0], unprojected_points_all[1])
    #ic(E, real_E / real_E[0,0] * E[0,0])
    retrieve_rt(E, dr, dt)

    #plot3d(transformed_points_all[0], indices)
    #plot2d(projected_points_all[0], window_width, window_width)

    s3d.shutdown()

