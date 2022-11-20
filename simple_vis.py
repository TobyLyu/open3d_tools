import open3d as o3d
import numpy as np
import yaml
import os, random, colorsys
from matplotlib import pyplot as plt

class Load:
    def __init__(self, path):
        self.path = path
        pass

    def reset(self):
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.remissions = np.zeros((0, 1), dtype=np.float32)
        # the first col of the color, R, contain semantic information
        self.colors = np.zeros((0, 3), dtype=np.float32)
        self.sem_label = np.zeros((0, 1), dtype=np.int)
        self.inst_label = np.zeros((0, 1), dtype=np.int)

    def open_everything(self, filename):
        self.reset()
        self.open_scan(filename=filename)
        self.open_label(filename=filename)
        # print(self.points, self.sem_label)

    def open_scan(self, filename):
        """Open the point cloud scan"""
        filename = 'velodyne/' + filename + '.bin'
        scan_path = os.path.join(self.path, filename)

        # check scan path is valid string
        if not isinstance(scan_path, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(scan_path))))

        # check extension is a laserscan
        if not any(scan_path.endswith(ext) for ext in ['.bin']):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(scan_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        self.points = scan[:, 0:3]  # get xyz
        self.remissions = scan[:, 3]  # get remission

    def open_label(self, filename):
        """ Open raw scan and fill in attributes
        """
        # check filename is string
        filename = 'labels/' + filename + '.label'
        label_path = os.path.join(self.path, filename)
        
        # check the path is valid string
        if not isinstance(label_path, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(label_path))))

        # check extension is a label
        if not any(label_path.endswith(ext) for ext in ['.label']):
            raise RuntimeError("Filename extension is not valid label file.")

        # if all goes well, open label
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))

        # set it
        self.__set_label__(label)

    def __set_label__(self, label):
        """ Set points for label not from file but from np
        """
        # check label makes sense
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")

        # only fill in attribute if the right size
        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label & 0xFFFF  # semantic label in lower half
            self.inst_label = label >> 16  # instance id in upper half
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        # sanity check
        assert ((self.sem_label + (self.inst_label << 16) == label).all())

class Color(Load):
    def __init__(self, config_path, data_path):
        super(Color, self).__init__(path=data_path)
        self.colors = None    
        self.open_config(filename=config_path)

    def open_config(self, filename):
        try:
            print("Opening config file %s" % filename)
            self.config = yaml.safe_load(open(filename, 'r'))
        except Exception as e:
            print(e)
            print("Error opening config yaml file.")
            quit()

    def set_colors(self):
        self.colors = np.zeros((0, 3), dtype=np.float32)
        config_color = self.config["color_map"]
        self.colors = np.zeros((len(self.points), 3), dtype=np.float32)
        for i in range(len(self.points)):
            self.colors[i, :] = np.array(config_color[self.sem_label[i]])/255

    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

class Rend(Color):
    def __init__(self, config_path, data_path):
        super(Rend, self).__init__(config_path=config_path, data_path = data_path)
        self.vis = o3d.visualization.Visualizer()
        self.pcd = o3d.geometry.PointCloud()
        self.open3d_config = None
    

    def create_window(self, name="Fig-1"):
        self.vis.create_window(window_name=name, width=1920, height=1080, left=10, top=10, visible=True)
        self.vis.add_geometry(self.pcd)
        # set view point
        ctr = self.vis.get_view_control()
        if self.open3d_config is None:
            ctr.set_front([ 0.0062476064943306157, -0.5890112666641909, 0.80810067142388731])
            ctr.set_lookat([3.8615300095966352, 0.59707870992582168, -1.1239333972284258])
            ctr.set_up([ -0.0053655146315808701, 0.80808506476267328, 0.58904137321604522 ])
            ctr.set_zoom(0.13999999999999962)
        self.vis.get_render_option().point_size = 2

    def update_window(self, start, end, save_path = []):
        for scan_id in range(start, end):
            # scan_id = 0
            filename = '{0:06d}'.format(scan_id)
            self.open_everything(filename=filename)
            self.set_colors()
            self.pcd.points = o3d.utility.Vector3dVector(self.points)
            self.pcd.colors = o3d.utility.Vector3dVector(self.colors)
            self.vis.update_geometry(self.pcd)
            self.vis.get_render_option().point_size = 2
            self.vis.poll_events()
            self.vis.update_renderer()
            # save
            self.save_window(scan_id = scan_id, save_path=save_path)

        self.vis.destroy_window()

    def save_window(self, scan_id, save_path=[]):
        if len(save_path):
            image = self.vis.capture_screen_float_buffer(True)
            save_path = os.path.join(save_path, '{0:06d}'.format(scan_id))
            plt.imsave(save_path, np.asarray(image), dpi=1)


    def animation_draw(self, start, end, save_path = []):
        # initialize
        filename = '{0:06d}'.format(start)
        self.open_everything(filename=filename)
        self.set_colors()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd.colors = o3d.utility.Vector3dVector(self.colors)
        self.create_window()
        # non-blocking render
        self.update_window(start=start, end=end, save_path=save_path)

    def single_draw(self, scan, save_path = []):
        # initialize
        filename = '{0:06d}'.format(scan)
        self.open_everything(filename=filename)
        self.set_colors()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd.colors = o3d.utility.Vector3dVector(self.colors)
        self.create_window()
        # render
        self.vis.run()
        # save
        self.save_window(scan_id=scan, save_path=save_path)
        self.vis.destroy_window()

if __name__ == '__main__':
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    dataset_path = "/media/lqy/My_Passport/Semantic_Point_Cloud/SemanticKitti/dataset/sequences/00/"   # xxxx/sequences/00/
    config_path = "/home/lqy/GitHub/RandLA-Net-master/utils/semantic-kitti.yaml"    # xxx/config.yaml
    save_path = ""      # xxxx/

    render = Rend(config_path=config_path, 
                                data_path=dataset_path)

    # ----------------choose your function--------------------#
    render.animation_draw(0, 100)
    # render.animation_draw(0, 100, save_path=save_path) # to enable save

    # render.single_draw(2)
    # render.single_draw(2, save_path=save_path)  # to enable save
     # ----------------------------------------------------------------#

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)