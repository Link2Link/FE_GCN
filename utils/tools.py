import numpy as np

def cuboid2box(cuboid):
    x, y, z = cuboid[0], cuboid[1], cuboid[2]
    xd, yd, zd = cuboid[3] / 2, cuboid[4] / 2, cuboid[5] / 2
    yaw = cuboid[6]
    R = np.array([np.sin(yaw), -np.cos(yaw), np.cos(yaw), np.sin(yaw)])
    pos = np.zeros([4, 2])
    pos[0] = [x + xd * R[0] - yd * R[1], y + xd * R[2] - yd * R[3]]
    pos[1] = [x + xd * R[0] + yd * R[1], y + xd * R[2] + yd * R[3]]
    pos[2] = [x - xd * R[0] + yd * R[1], y - xd * R[2] + yd * R[3]]
    pos[3] = [x - xd * R[0] - yd * R[1], y - xd * R[2] - yd * R[3]]
    return pos

def point2line(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    A = y2 - y1
    B = x1 - x2
    C = -(x1 * (y2 - y1) + y1 * (x1 - x2))
    return np.array([A, B, C])


def box_check(points, cuboid, threshold=0.7):
    vertex = cuboid2box(cuboid)
    vertex = np.concatenate((vertex, np.ones([4,1])), axis=1)
    vertex_m = np.mean(vertex, axis=0)

    L1 = point2line(vertex[0], vertex[1])
    L2 = point2line(vertex[1], vertex[2])
    L3 = point2line(vertex[2], vertex[3])
    L4 = point2line(vertex[3], vertex[0])


    L1 *= np.sign(vertex_m @ L1)
    L2 *= np.sign(vertex_m @ L2)
    L3 *= np.sign(vertex_m @ L3)
    L4 *= np.sign(vertex_m @ L4)

    points2d = points[:, :2]
    points2d = np.concatenate((points2d, np.ones([len(points2d), 1])), axis=1)
    index = (points2d @ L1 > 0) & (points2d @ L2 > 0) & (points2d @ L3 > 0) & (points2d @ L4 > 0)
    num_inbox = np.sum(index)
    if num_inbox/len(points) > threshold:
        return True
    else:
        return False


