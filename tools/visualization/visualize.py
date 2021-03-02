import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from vispy.color import Colormap
from vispy.color import ColorArray

color_dict = {  'white':                '#FFFFFF',
                'aqua':                 '#00FFFF',
                'red':                  '#FF0000',
                'yellow':                 '#FF9912',
                }

def rotation(center, size, angle, format='HWL', high_bias=True):
    H, W, L = size[format.find('H')], size[format.find('W')], size[format.find('L')]
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    box_2d = np.array([[L/2, W/2], [-L/2, W/2], [-L/2, -W/2], [L/2, -W/2]])
    box_2d = box_2d@R.transpose()
    box_3d = np.zeros([8, 3])
    if high_bias:
        box_3d[:4, :2], box_3d[:4, 2] = box_2d, H
        box_3d[-4:, :2], box_3d[-4:, 2] = box_2d, 0
    else:
        box_3d[:4, :2], box_3d[:4, 2] = box_2d, H/2
        box_3d[-4:, :2], box_3d[-4:, 2] = box_2d, -H/2

    return box_3d + center




def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)


color_map = [color(color_dict[key]) for key in list(color_dict.keys())]
color_map = np.array(color_map)




class visualization:
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self._draw)
        self.grid = self.canvas.central_widget.add_grid()

        self.init_featureshow()

        # visuals.XYZAxis(parent=self.view.scene)
        # visuals.XYZAxis(parent=self.view2.scene)
        self.cm = Colormap(ColorArray(['#9933FA',
                                       '#FFFFFF',
                                       '#3D9140',
                                       '#00FFFF',
                                       '#FF7D40',
                                       '#03A89E',
                                       ]))

    def init_pointcloud(self):
        self.view = vispy.scene.widgets.ViewBox(bgcolor='white', parent=self.canvas.scene)
        self.view2 = vispy.scene.widgets.ViewBox(bgcolor='white', parent=self.canvas.scene)
        self.view3 = vispy.scene.widgets.ViewBox(bgcolor='white', parent=self.canvas.scene)

        self.grid.add_widget(self.view, 0, 0)
        self.grid.add_widget(self.view2, 0, 1)
        self.grid.add_widget(self.view3, 0, 2)

        self.vis = visuals.Markers()
        self.vis2 = visuals.Markers()
        self.vis3 = visuals.Markers()

        self.box = visuals.Line()
        self.box2 = visuals.Line()
        self.box3 = visuals.Line()

        self.view.camera = 'fly'  # 'panzoom' #'turntable'
        self.view2.camera = self.view.camera  # sharing camera
        self.view3.camera = self.view.camera  # sharing camera

        self.view.add(self.vis)
        self.view.add(self.box)
        self.view2.add(self.vis2)
        self.view2.add(self.box2)
        self.view3.add(self.vis3)
        self.view3.add(self.box3)

    def init_featureshow(self):
        self.view = vispy.scene.widgets.ViewBox(bgcolor='black', parent=self.canvas.scene)
        self.view2 = vispy.scene.widgets.ViewBox(bgcolor='white', parent=self.canvas.scene)
        self.view3 = vispy.scene.widgets.ViewBox(bgcolor='white', parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)
        self.grid.add_widget(self.view2, 0, 1)
        self.grid.add_widget(self.view3, 0, 2)

        self.vis = visuals.Markers()
        self.vis2 = visuals.Image()
        self.vis3 = visuals.Image()

        self.view.camera = 'turntable'  # 'panzoom' #'turntable'
        self.view2.camera = 'panzoom'
        self.view3.camera = self.view2.camera  # sharing camera

        self.view.add(self.vis)
        self.view2.add(self.vis2)
        self.view3.add(self.vis3)





    def key_press(self, event):
        self.canvas.events.key_press.block()
        if event.key == 'N':
            self.press_N()
        elif event.key == 'B':
            self.press_B()
        elif event.key == 'Q' or event.key == 'Escape':
            self.canvas.close()
            vispy.app.quit()
        else:
            self.press_butten(event.key)

    def press_butten(self, key):
        pass

    def press_N(self):
        raise NotImplementedError

    def press_B(self):
        raise NotImplementedError

    def _draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def draw_semseg(self, points, label, flag=0):
        face_color = color_map[label].astype(np.float) / 255
        if flag==0:
            self.vis.set_data(points, edge_color=None, face_color=face_color, size=3)
        else:
            self.vis2.set_data(points, edge_color=None, face_color=face_color, size=3)

    def draw_points(self, points, flag=0, D2=False):
        if D2:
            points[:, 2] = 0

        distance = np.linalg.norm(points, axis=1)
        distance -= np.min(distance)
        distance /= np.max(distance)
        if flag == 0:
            self.vis.set_data(points, edge_color=None, face_color=self.cm[distance], size=5, symbol='square')
            print(distance)
        elif flag == 1:
            self.vis2.set_data(points, edge_color=None, face_color=self.cm[distance], size=3)
        else:
            self.vis3.set_data(points, edge_color=None, face_color=self.cm[distance], size=3, symbol='diamond')

    def draw_bbox(self, cuboids, label, flag=0, dataset='kitti', D2=False):
        if dataset == 'kitti':
            format = 'HWL'
            angle_bias = 1
            high_bias = True
        else:
            format='LWH'
            angle_bias = 1
            high_bias = False

        format = 'LWH'
        high_bias = False
        angle_bias = 0

        face_color = color_map[label].astype(np.float) / 255
        color_array = np.empty([0, 3])

        cube = np.expand_dims(cuboids, 0) if cuboids.ndim == 1 else cuboids
        num = len(cube)
        center = cube[:, :3]
        size = cube[:, 3:6]
        angle = cube[:, 6] - angle_bias * np.pi/2
        pos = np.zeros([num, 8, 3])
        for i in range(num):
            pos[i] = rotation(center[i], size[i], angle[i], format=format, high_bias=high_bias)
        boxes = np.empty([0, 3])
        for box in pos:
            boxes = np.concatenate((boxes, box), axis=0)
        box_c = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]])
        connect = np.empty([0, 2], dtype=np.int)
        for i in range(num):
            connect = np.concatenate((connect, box_c + 8*i), axis=0)
            label_color = np.ones([8, 3]) * face_color[i]
            color_array = np.concatenate((color_array, label_color))

        if D2:
            boxes[:, 2] = 0

        if flag == 0:
            self.box.set_data(pos=boxes, face_color=color_array, width=3, connect=connect)
        elif flag == 1:
            self.box2.set_data(pos=boxes, color=color_array, width=3, connect=connect)
        else:
            self.box3.set_data(pos=boxes, color=color_array, width=3, connect=connect)

    # def draw_text(self, text, label, font_size, boxes):
    #     face_color = color_map[label].astype(np.float) / 255
    #     color_array = np.empty([0, 3])
    #     cube = np.expand_dims(boxes, 0) if boxes.ndim == 1 else boxes
    #     center = cube[:, :3]
    #     self.text.set_data(text=text, color=color_array, font_size=font_size, pos=center)

    def draw_img(self, img, flag=1):
        if flag == 1:
            self.vis2.set_data(img)
        elif flag == 2:
            self.vis3.set_data(img)