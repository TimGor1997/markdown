# Kitti Dataset核心代码
## Kitti Dataset基类
```python
class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split = 'train'):
        self.split = split
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object', 'testing' if is_test else 'training')

        split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', split + '.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.image_idx_list.__len__()

        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # Don't need to permute while using grid_sample
        self.image_hw_with_padding_np = np.array([1280., 384.])

    def get_image_rgb_with_normal(self, idx):
        return imback   #（H, W, 3)
    def get_image_shape_with_padding(self, idx=0):
        return 384, 1280, 3
    def get_image_shape(self, idx):
        return height, width, 3
    def get_lidar(self, idx):
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    def get_calib(self, idx):
        return calibration.Calibration(calib_file)
    def get_label(self, idx):
        return kitti_utils.get_objects_from_label(label_file)
    def get_road_plane(self, idx):
        return plane
    def __len__(self):
        raise NotImplementedError
    def __getitem(self, item):
        raise NotImplementedError
```
## KittiRCNNDataset核心代码
```python
class KittiRCNNDataset(KittiDataset):
    def __init__(self, root_dir, npoints = 16384, split = 'train', classes = 'Car', mode = 'TRAIN',
                 random_select = True,
                 logger = None, rcnn_training_roi_dir = None, rcnn_training_feature_dir = None,
                 rcnn_eval_roi_dir = None,
                 rcnn_eval_feature_dir = None, gt_database_dir = None):
        super().__init__(root_dir = root_dir, split = split)
        self.classes = ('Background', 'Car')
        self.num_class = self.classes.__len__()
        self.npoints = npoints
        self.sample_id_list = []
        self.random_select = random_select
        self.logger = logger
        if split == 'train_aug':
            self.aug_label_dir = os.path.join(aug_scene_root_dir, 'training', 'aug_label')
            self.aug_pts_dir = os.path.join(aug_scene_root_dir, 'training', 'rectified_data')
        else:
            self.aug_label_dir = os.path.join(aug_scene_root_dir, 'training', 'aug_label')
            self.aug_pts_dir = os.path.join(aug_scene_root_dir, 'training', 'rectified_data')

        # for rcnn training
        self.rcnn_training_bbox_list = []
        self.rpn_feature_list = {}
        self.pos_bbox_list = []
        self.neg_bbox_list = []
        self.far_neg_bbox_list = []
        self.rcnn_eval_roi_dir = rcnn_eval_roi_dir
        self.rcnn_eval_feature_dir = rcnn_eval_feature_dir
        self.rcnn_training_roi_dir = rcnn_training_roi_dir
        self.rcnn_training_feature_dir = rcnn_training_feature_dir
        self.gt_database = None
        self.mode = mode

        if cfg.RPN.ENABLED:
            if mode == 'TRAIN':
                self.preprocess_rpn_training_data()
            else:
                self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
                self.logger.info('Load testing samples from %s' % self.imageset_dir)
                self.logger.info('Done: total test samples %d' % len(self.sample_id_list))
        elif cfg.RCNN.ENABLED:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
            self.logger.info('Load testing samples from %s' % self.imageset_dir)
            self.logger.info('Done: total test samples %d' % len(self.sample_id_list))
    def preprocess_rpn_training_data(self):
        """
        Discard samples which don't have current classes, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        self.logger.info('Loading %s samples from %s ...' % (self.mode, self.label_dir))
        for idx in range(0, self.num_sample):
            sample_id = int(self.image_idx_list[idx])
            obj_list = self.filtrate_objects(self.get_label(sample_id))
            if len(obj_list) == 0:
                continue
            self.sample_id_list.append(sample_id)
        self.logger.info('Done: filter %s results: %d / %d\n' % (self.mode, len(self.sample_id_list),
                                                                 len(self.image_idx_list)))
    def get_label(self, idx):
        return kitti_utils.get_objects_from_label(label_file)
    def get_image(self, idx):
        return super().get_image(idx % 10000)
    def get_image_shape(self, idx):
        return super().get_image_shape(idx % 10000)
    def get_calib(self, idx):
        return super().get_calib(idx % 10000)
    def get_road_plane(self, idx):
        return super().get_road_plane(idx % 10000)
    
    @staticmethod
    def get_rpn_features(rpn_feature_dir, idx):
        rpn_feature_file = os.path.join(rpn_feature_dir, '%06d.npy' % idx)
        rpn_xyz_file = os.path.join(rpn_feature_dir, '%06d_xyz.npy' % idx)
        rpn_intensity_file = os.path.join(rpn_feature_dir, '%06d_intensity.npy' % idx)
        if cfg.RCNN.USE_SEG_SCORE:  # False
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_rawscore.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
            rpn_seg_score = torch.sigmoid(torch.from_numpy(rpn_seg_score)).numpy()
        else:
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_seg.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
        return np.load(rpn_xyz_file), np.load(rpn_feature_file), np.load(rpn_intensity_file).reshape(-1), rpn_seg_score

    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes
        if self.mode == 'TRAIN' and cfg.INCLUDE_SIMILAR_TYPE:  # True
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')
            if 'Pedestrian' in self.classes:  # or 'Cyclist' in self.classes:
                type_whitelist.append('Person_sitting')

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:
                continue
            if self.mode == 'TRAIN' and cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(obj.pos) is False):
                continue
            valid_obj_list.append(obj)
        return valid_obj_list
    
    @staticmethod
    def filtrate_dc_objects(obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type in ['DontCare']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    @staticmethod
    def check_pc_range(xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range, y_range, z_range = cfg.PC_AREA_SCOPE
        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        if cfg.PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = cfg.PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    def __len__(self):
        if cfg.RPN.ENABLED:
            return len(self.sample_id_list)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                return len(self.sample_id_list)
            else:
                return len(self.image_idx_list)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if cfg.LI_FUSION.ENABLED:
            return self.get_rpn_with_li_fusion(index)

        if cfg.RPN.ENABLED:
            return self.get_rpn_sample(index)
        # elif cfg.RCNN.ENABLED:
        #     if self.mode == 'TRAIN':
        #         if cfg.RCNN.ROI_SAMPLE_JIT:
        #             return self.get_rcnn_sample_jit(index)
        #         else:
        #             return self.get_rcnn_training_sample_batch(index)
        #     else:
        #         return self.get_proposal_from_file(index)
        # else:
        # raise NotImplementedError
