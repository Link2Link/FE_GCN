import vispy
from vispy.scene import visuals, SceneCanvas
from utils import color
import numpy as np

class visualization:
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self._draw)
        self.grid = self.canvas.central_widget.add_grid()

        self.view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
        self.view2 = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)

        self.grid.add_widget(self.view, 0, 0)
        self.grid.add_widget(self.view2, 0, 1)

        self.vis = visuals.Markers()
        self.vis2 = visuals.Markers()
        self.box = visuals.Line()
        self.box2 = visuals.Line()

        self.view.camera = 'turntable'
        self.view2.camera = self.view.camera    # sharing camera

        self.view.add(self.vis)
        self.view.add(self.box)
        self.view2.add(self.vis2)
        self.view2.add(self.box2)
        visuals.XYZAxis(parent=self.view.scene)
        visuals.XYZAxis(parent=self.view2.scene)

    def key_press(self, event):
        self.canvas.events.key_press.block()
        if event.key == 'N':
            self.press_N()
        elif event.key == 'B':
            self.press_B()
        elif event.key == 'Q' or event.key == 'Escape':
            self.canvas.close()
            vispy.app.quit()

    def press_N(self):
        raise NotImplementedError

    def press_B(self):
        raise NotImplementedError

    def _draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def draw_semseg(self, points, label, flag=0):
        face_color = color.color_map[label].astype(np.float) / 255
        if flag==0:
            self.vis.set_data(points, edge_color=None, face_color=face_color, size=3)
        else:
            self.vis2.set_data(points, edge_color=None, face_color=face_color, size=3)

    def draw_bbox(self, cuboids, flag=0):
        cube = np.expand_dims(cuboids, 0) if cuboids.ndim == 1 else cuboids
        num = len(cube)
        x_c = cube[:, 0]
        y_c = cube[:, 1]
        z_c = cube[:, 2]
        x_d = cube[:, 3]
        y_d = cube[:, 4]
        z_d = cube[:, 5]
        yaw = cube[:, 6]

        pos = np.zeros([num, 8, 3])
        for i in range(num):
            if num == 1:
                x, y, z = x_c, y_c, z_c
                xd, yd, zd = x_d / 2, y_d / 2, z_d / 2
                R = np.array([np.sin(yaw), -np.cos(yaw), np.cos(yaw), np.sin(yaw)])
            else:
                x, y, z = x_c[i], y_c[i], z_c[i]
                xd, yd, zd = x_d[i]/2, y_d[i]/2, z_d[i]/2
                R = np.array([np.sin(yaw[i]), -np.cos(yaw[i]), np.cos(yaw[i]), np.sin(yaw[i])])

            pos[i, 0] = [x + xd * R[0] - yd * R[1], y + xd * R[2] - yd * R[3], z - zd]
            pos[i, 1] = [x + xd * R[0] + yd * R[1], y + xd * R[2] + yd * R[3], z - zd]
            pos[i, 2] = [x - xd * R[0] + yd * R[1], y - xd * R[2] + yd * R[3], z - zd]
            pos[i, 3] = [x - xd * R[0] - yd * R[1], y - xd * R[2] - yd * R[3], z - zd]

            pos[i, 4] = [x + xd * R[0] - yd * R[1], y + xd * R[2] - yd * R[3], z + zd]
            pos[i, 5] = [x + xd * R[0] + yd * R[1], y + xd * R[2] + yd * R[3], z + zd]
            pos[i, 6] = [x - xd * R[0] + yd * R[1], y - xd * R[2] + yd * R[3], z + zd]
            pos[i, 7] = [x - xd * R[0] - yd * R[1], y - xd * R[2] - yd * R[3], z + zd]
        boxes = np.empty([0, 3])
        for box in pos:
            boxes = np.concatenate((boxes, box), axis=0)
        box_c = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]])
        connect = np.empty([0, 2], dtype=np.int)
        for i in range(num):
            connect = np.concatenate((connect, box_c + 8*i), axis=0)

        if flag == 0:
            self.box.set_data(pos=boxes, color=[1, 0, 0], width=1, connect=connect)
        else:
            self.box2.set_data(pos=boxes, color=[1, 0, 0], width=1, connect=connect)


