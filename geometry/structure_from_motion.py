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
       a * Na.T^-1 * E * Nb^-1 * b = 0,
       a * P * b = 0.
    So the P found by this procedure should be convertible to the
    matrix E by:
       E = Na.T @ P @ Nb

    :param pts_a:
    :param pts_b:
    :return:
    """

    """
    rnda = np.random.normal(scale=0.002, size=(pts_a.shape[0], 2))
    rndb = np.random.normal(scale=0.002, size=(pts_a.shape[0], 2))
    ic(np.mean(np.abs(rnda)), np.mean(np.abs(rndb)))
    ic(np.max(pts_a[:, :2], 0) - np.min(pts_a[:, :2], 0))
    pts_a[:, :2] += rnda
    pts_b[:, :2] += rndb
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
    E_handedness = np.dot(np.cross(E_raw[:, 0], E_raw[:, 1]), E_raw[:, 2])

    dbg = lambda m: np.dot(np.cross(m[:, 0], m[:, 1]), m[:, 2])
    if E_handedness < 0 and False:
        """
        By expanding ((C0 cross C1) dot C2) for [t]x=[C0,C1,C2] one will find that this
        value (which is related to the handedness of the operation) should be 0.
        
        Due to unknown reasons yet (possibly inaccuracies in the location of keypoints),
        E can have (...)
        """
        # TODO where should this be done?
        E_raw = E_raw @ np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ])
        ic(dbg(E_raw))
        pass


    # Make E have rank 2
    ue, se, vhe = scipy.linalg.svd(E_raw)
    E = ue @ np.diag([se[0], se[1], 0]) @ vhe

    # How does E compare to real_E in terms of ||A@E||?
    ic(np.linalg.norm(a_times_b @ E_raw.reshape((9,1))))
    ic(np.linalg.norm(a_times_b @ E.reshape((9,1))))
    real_E_n = np.linalg.inv(normalization_a.T) @ real_E @ np.linalg.inv(normalization_b)
    ic(np.linalg.norm(a_times_b @ real_E_n.reshape((9,1))))
    exit()

    # Remove normalization transform
    E = normalization_a.T @ E @ normalization_b

    # Show that a.T @ E @ b = 0 (ideally; may not occur in real life)
    #ic('a.T @ estimated E @ b', mean_std(np.abs([
    #    pt_a[np.newaxis, :] @ E @ pt_b[:, np.newaxis]
    #    for pt_a, pt_b in zip(pts_a, pts_b)
    #])))

    # Show that a.T @ real_E @ b = 0
    #ic('a.T @ real E @ b', mean_std(np.abs([
    #    pt_a[np.newaxis, :] @ real_E @ pt_b[:, np.newaxis]
    #    for pt_a, pt_b in zip(pts_a, pts_b)
    #])))

    # Plot point clouds and epipolar lines
    """
    # Scatter plot point clouds
    #pts_a=pts_a_n
    #pts_b=pts_b_n
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
    origin_a_in_b = homognorm(-real_dr.T @ real_dt)
    ax[1].scatter(*origin_a_in_b.flatten()[:2], s=50, marker='*', color=(0,0,0))

    # Adjust plot style
    rng = np.max(np.abs(np.concatenate([pts_a[:, :2], pts_b[:, :2]], axis=0)))
    for axi in ax:
        axi.set_aspect(1)
        axi.set_xlim([-3 * rng, 3 * rng])
        axi.set_ylim([-3 * rng, 3 * rng])
        axi.invert_yaxis()

    plt.show()
    #"""

    return E


def refine_essential():
    """
    pts_a @ E @ pts_b = 0.
    We can rewrite E as a 9 row vector. With A such that
    A_[i, :] = (pts_a_i @ pts_b_i.T).flat
             = [x_a*x_b x_a*y_b x_a y_a*x_b y_a*y_b y_a x_b y_b 1],
    we equivalently have A @ E' = 0.

    Since E = [t] x @ R, if we have a reliable estimate of t, then we want
    to find R such that A @ ([t]x @ R)' = 0. Here,
    ([t]x @ R)' = [[t]x00*R00 + [t]x01*R10 + [t]x02*R20
                   [t]x00*R01 + [t]x01*R11 + [t]x02*R21
                   [t]x00*R02 + [t]x01*R12 + [t]x02*R22

                   [t]x10*R00 + [t]x11*R10 + [t]x12*R20
                   [t]x10*R01 + [t]x11*R11 + [t]x12*R21
                   [t]x10*R02 + [t]x11*R12 + [t]x12*R22

                   [t]x20*R00 + [t]x21*R10 + [t]x22*R20
                   [t]x20*R01 + [t]x21*R11 + [t]x22*R21
                   [t]x20*R02 + [t]x21*R12 + [t]x22*R22]
    If we write ([t]x @ R)' = T @ R', with R'=R.flat, then
    R'= [r00    r01    r02    r10    r11    r12    r20    r21    r22]
    T = [[t]x00 0      0      [t]x01 0      0      [t]x02 0      0
         0      [t]x00 0      0      [t]x01 0      0      [t]x02 0
         0      0      [t]x00 0      0      [t]x01 0      0      [t]x02
         [t]x10 0      0      [t]x11 0      0      [t]x12 0      0
         0      [t]x10 0      0      [t]x11 0      0      [t]x12 0
         0      0      [t]x10 0      0      [t]x11 0      0      [t]x12
         [t]x20 0      0      [t]x21 0      0      [t]x22 0      0
         0      [t]x20 0      0      [t]x21 0      0      [t]x22 0
         0      0      [t]x20 0      0      [t]x21 0      0      [t]x22]

    Hartley proposes that we find the f = T @ R' that minimizes ||A@f|| subject
    to f=T@R' to a known T and free R', instead of minimizing ||A@T@R'||. The idea
    is to avoid solutions that lie in the null space of T.

    :return:
    """
    pass

def get_rt(E, pts_a, pts_b):
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
    after Rt is applied). This means:
      E = U @ S @ V
        = Rt.T @ R90 @ S @ K.T @ K @ Rt @ r
        = Rt.T @ R90 @ K.T @ S @ K @ Rt @ r
    Note that S @ K.T = K.T @ S due to K being a 2d rotation around Z and S being a uniform
    scale in XY.

    Thus, U=Rt.T @ R90 @ K.T=Rt.T @ K.T @ R90 (because 2d rotations are commutative) and
    V=K @ Rt @ r. We can finally retrieve r:
      r = Rt.T @ K.T @ V.
    Since Rt.T = U @ K @ R90.T,
      r = U @ K @ R90.T @ K.T @ V (by replacing Rt.T in the r identity above),
        = U @ R90.T @ K @ K.T @ V (R90.T and K are commutative as they rotate around Z),
        = U @ R90.T @ V
    """
    u, s, vh = scipy.linalg.svd(E)
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
    Z=np.array([0,0,1])
    Rt = R.from_rotvec(np.arccos(np.dot(normalize(real_dt).flatten(), Z))*normalize(np.cross(real_dt.flatten(), Z))).as_matrix()
    K = real_dr @ real_dr.T @ Rt.T
    righthandedness = lambda m: np.dot(np.cross(m[:, 0], m[:, 1]), m[:, 2])
    ic(righthandedness(E))
    ic(righthandedness(Rt))
    ic(righthandedness(K))
    ic(righthandedness(u), righthandedness(vh))
    r90 = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ])
    # TODO
    translation = -u[:, 2:3]
    rotation = u @ r90.T @ vh
    # TODO understand why found matrix can be left handed
    #rotation[:, 2] = np.cross(rotation[:, 0], rotation[:, 1])

    # How do our results compare to the groundtruth?
    ic(normalize(real_dt), translation)
    ic(real_dr, rotation)
    ic(real_dr.T @ rotation)
    ic('Axis distances in degrees',
       np.rad2deg(np.arccos(np.diag(real_dr.T @ rotation))))
    exit()

    return rotation, translation


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
    # TODO camera mat seems to be not affecting image
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

    # What are the statistical properties of the error?
    #err = unprojected_points_all[0] - unprojected_points_all_noround[0]
    #ic(err, np.mean(err, 0), np.std(err, 0))
    #plt.hist(err[:, 0], bins=40)
    #plt.hist(err[:, 1], bins=40)
    #plt.show()

    #E = get_essential(unprojected_points_all_noround[0], unprojected_points_all_noround[1])
    E = get_essential(unprojected_points_all[0], unprojected_points_all[1])
    est_dr, est_dt = get_rt(E, unprojected_points_all[0], unprojected_points_all[1])

    #plot3d(transformed_points_all[0], indices)
    #plot2d(projected_points_all[0], window_width, window_width)

    s3d.shutdown()

