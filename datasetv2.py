import numpy as np
import os
import cv2
import math
import parse
import pandas as pd

from PIL import Image
from tdqm import tqdm
from torch.utils.data import Dataset, IterableDataset
from utils.general import get_impact_dirs, get_session_median, HEIGHT, WIDTH, SIGMA, IMG_FORMAT

data_dir = 'data'

class GolfBall_Trajectory_Dataset(Dataset):
    def __init__(self,
        root_dir=data_dir,
        split='train',
        seq_len=8,
        sliding_step=1,
        data_mode='heatmap',
        bg_mode='',
        frame_alpha=-1,
        impact_dir=None,
        frame_arr=None,
        pred_dict=None,
        padding=False,
        debug=False,
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        SIGMA=SIGMA,
        median=None
    ):
        """ Initialize the dataset

            Args:
                root_dir (str): File path of root directory of the dataset
                split (str): Split of the dataset, 'train', 'test' or 'val'
                seq_len (int): Length of the input sequence
                sliding_step (int): Sliding step of the sliding window during the generation of input sequences
                data_mode (str): Data mode
                    Choices:
                        - 'heatmap':Return TrackNet input data
                        - 'coordinate': Return InpaintNet input data
                bg_mode (str): Background mode
                    Choices:
                        - '': Return original frame sequence
                        - 'subtract': Return the difference frame sequence
                        - 'subtract_concat': Return the frame sequence with RGB and difference frame channels
                        - 'concat': Return the frame sequence with background as the first frame
                frame_alpha (float): Frame mixup alpha
                impact_dir (str): Impact frame directory
                frame_arr (numpy.ndarray): Frame sequence for TrackNet inference
                pred_dict (Dict): Prediction dictionary for InpaintNet inference
                    Format: {'X': x_pred (List[int]),
                             'Y': y_pred (List[int]),
                             'Visibility': vis_pred (List[int]),
                             'Inpaint_Mask': inpaint_mask (List[int]),
                             'Img_scaler': img_scaler (Tuple[int]),
                             'Img_shape': img_shape (Tuple[int])}
                padding (bool): Padding the last frame if the frame sequence is shorter than the input sequence
                debug (bool): Debug mode
                HEIGHT (int): Height of the image for input.
                WIDTH (int): Width of the image for input.
                SIGMA (int): Sigma of the Gaussian heatmap which control the label size.
                median (numpy.ndarray): Median image
        """

        assert split in ['train', 'test', 'val'], f'Invalid split: {split}, should be train, test or val'
        assert data_mode in ['heatmap', 'coordinate'], f'Invalid data_mode: {data_mode}, should be heatmap or coordinate'
        assert bg_mode in ['', 'subtract', 'subtract_concat', 'concat'], f'Invalid bg_mode: {bg_mode}, should be "", subtract, subtract_concat or concat'

        # Image size
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

        # Gaussian heatmap parameters
        self.mag = 1
        self.sigma = SIGMA

        self.root_dir = root_dir
        self.split = split if impact_dir is None else self._get_split(impact_dir)
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data_mode = data_mode
        self.bg_mode = bg_mode
        self.frame_alpha = frame_alpha

        # Data for inference
        self.frame_arr = frame_arr
        self.pred_dict = pred_dict
        self.padding = padding and self.sliding_step == self.seq_len

        # Initialize the input data
        if self.frame_arr is not None:
            # For TrackNet inference
            assert self.data_mode == 'heatmap', f'Invalid data_mode: {self.data_mode}, frame_arr only for heatmap mode' 
            self.data_dict, self.img_config = self._gen_input_from_frame_arr()

            # background mode
            if self.bg_mode:
                if median is None:
                    # Take the first image of the frame sequence as the background image
                    median = np.median(self.frame_arr, 0)
                if self.bg_mode == 'concat':
                    # concat the median image as the first frame
                    median = Image.fromarray(median.astype('uint8'))
                    median = np.array(median.resize(size=(self.WIDTH, self.HEIGHT)))
                    self.median = np.moveaxis(median, -1, 0)
                else:
                    self.median = median
        
        elif self.pred_dict is not None:
            # For InpaintNet inference
            assert self.data_mode == 'coordinate', f'Invalid data_mode: {self.data_mode}, pred_dict only for coordinate mode'
            self.data_dict, self.img_config = self._gen_input_from_pred_dict()
        else:
            # Generate rally image configuration file
            #self.rally_dict = self._get_rally_dict()
            self.impact_dict = self._get_impact_frame_dict()

            img_config_file = os.path.join(self.root_dir, f'img_config_{self.HEIGHT}x{self.WIDTH}_{self.split}.npz')
            if not os.path.exists(img_config_file):
                self._gen_impact_img_config_file(img_config_file)
                # self._gen_rally_img_congif_file(img_config_file)
            img_config = np.load(img_config_file)
            self.img_config = {key: img_config[key] for key in img_config.keys()}
            
            # For training and evaluation
            if impact_dir is not None:
                # Impact frame based
                self.data_dict = self._gen_input_from_impact_frame_dir(impact_dir)
            else:
                # Split based
                # Generate and load input file 
                input_file = os.path.join(self.root_dir, f'data_l{self.seq_len}_s{self.sliding_step}_{self.data_mode}_{self.split}.npz')
                if not os.path.exists(input_file):
                    self._gen_input_file(file_name=input_file)
                data_dict = np.load(input_file)
                self.data_dict = {key: data_dict[key] for key in data_dict.keys()}
            if debug:
                num_data = 256
                for key in self.data_dict.keys():
                    self.data_dict[key] = self.data_dict[key][:num_data]

    def _get_impact_dict(self) -> dict:
        impact_dirs = get_impact_dirs(self.root_dir, self.split)
        impact_dict = {'i2p':{i: os.path.join(self.root_dir, impact_dir) for i, impact_dir in enumerate(impact_dirs)},
                       'p2i':{os.path.join(self.root_dir, impact_dir): i for i, impact_dir in enumerate(impact_dirs)}}
        return impact_dict
    
    def _get_impact_i(self, impact_dir) -> int | None:
        if impact_dir not in self.impact_dict['p2i'].keys():
            return None
        else:
            return self.impact_dict['p2i'][impact_dir]
        
    def _get_split(self, impact_dir) -> str:
        file_format_str = os.path.join(self.root_dir, '{}', 'impact{}')
        split, _ = parse.parse(file_format_str, impact_dir)
        return split
    
    def _gen_impact_img_config_file(self, file_name) -> None:
        img_scaler = [] # (num_impact, 2)
        img_shape = [] # (num_impact, 2)

        for impact_i, impact_dir in tqdm(self.impact_dict['i2p'].items()):
            w, h = Image.open(os.path.join(impact_dir, f'0.{IMG_FORMAT}')).size
            w_scaler, h_scaler = w / self.WIDTH, h / self.HEIGHT
            img_scaler.append((w_scaler, h_scaler))
            img_shape.append((w, h))
        
        np.savez(file_name, img_scaler=img_scaler, img_shape=img_shape)

    def _gen_input_file(self, file_name) -> None:
        print('Generate input file...')
        
        if self.data_mode == 'heatmap':
            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            frame_file = np.array([]).reshape(0, self.seq_len)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            # Generate input sequences from each impact
            for impact_i, impact_dir in tqdm(self.impact_dict['i2p'].items()):
                data_dict = self._gen_input_from_impact_dir(impact_dir)
                id = np.concatenate((id, data_dict['id']), axis=0)
                frame_file = np.concatenate((frame_file, data_dict['frame_file']), axis=0)
                coor = np.concatenate((coor, data_dict['coor']), axis=0)
                vis = np.concatenate((vis, data_dict['vis']), axis=0)
            
            np.savez(file_name, id=id, frame_file=frame_file, coor=coor, vis=vis)
        else:
            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            # Generate input sequences from each impact
            for impact_i, impact_dir in tqdm(self.impact_dict['i2p'].items()):
                data_dict = self._gen_input_from_impact_dir(impact_dir)
                id = np.concatenate((id, data_dict['id']), axis=0)
                coor = np.concatenate((coor, data_dict['coor']), axis=0)
                coor_pred = np.concatenate((coor_pred, data_dict['coor_pred']), axis=0)
                vis = np.concatenate((vis, data_dict['vis']), axis=0)
                pred_vis = np.concatenate((pred_vis, data_dict['pred_vis']), axis=0)
                inpaint_mask = np.concatenate((inpaint_mask, data_dict['inpaint_mask']), axis=0)
            
            np.savez(file_name, id=id, coor=coor, coor_pred=coor_pred,
                     vis=vis, pred_vis=pred_vis, inpaint_mask=inpaint_mask)


    def _gen_input_from_impact_dir(self, impact_dir) -> dict:
        """ Generate input sequences from an impact directory. """

        impact_i = self._get_impact_i(impact_dir)
        
        file_format_str = os.path.join('{}', 'frame', '{}')
        session_dir, impact_id = parse.parse(file_format_str, impact_dir)
        
        if self.data_mode == 'heatmap':
            # Read label csv file
            if 'test' in impact_dir:
                csv_file = os.path.join(session_dir, 'corrected_csv', f'{impact_id}_ball.csv')
            else:
                csv_file = os.path.join(session_dir, 'csv', f'{impact_id}_ball.csv')
            
            assert os.path.exists(csv_file), f'{csv_file} does not exist.'
            label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)

            f_file = np.array([os.path.join(impact_dir, f'{f_id}.{IMG_FORMAT}') for f_id in label_df['Frame']])
            x, y, v = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])

            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            frame_file = np.array([]).reshape(0, self.seq_len)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            
            # Sliding on the frame sequence
            last_idx = -1
            for i in range(0, len(f_file), self.sliding_step):
                tmp_idx, tmp_frames, tmp_coor, tmp_vis = [], [], [], []
                # Construct a single input sequence
                for f in range(self.seq_len):
                    if i+f < len(f_file):
                        tmp_idx.append((impact_i, i+f))
                        tmp_frames.append(f_file[i+f])
                        tmp_coor.append((x[i+f], y[i+f]))
                        tmp_vis.append(v[i+f])
                        last_idx = i+f
                    else:
                        # Padding the last sequence if imcompleted
                        if self.padding:
                            tmp_idx.append((impact_i, last_idx))
                            tmp_frames.append(f_file[last_idx])
                            tmp_coor.append((x[last_idx], y[last_idx]))
                            tmp_vis.append(v[last_idx])
                        else:
                            break
                
                # Append the input sequence
                if len(tmp_frames) == self.seq_len:
                    assert len(tmp_frames) == len(tmp_coor) == len(tmp_vis),\
                    f'Length of frames, coordinates and visibilities are not equal.'
                    id = np.concatenate((id, [tmp_idx]), axis=0)
                    frame_file = np.concatenate((frame_file, [tmp_frames]), axis=0)
                    coor = np.concatenate((coor, [tmp_coor]), axis=0)
                    vis = np.concatenate((vis, [tmp_vis]), axis=0)
            
            return dict(id=id, frame_file=frame_file, coor=coor, vis=vis)
        else:
            # Read predicted csv file
            pred_csv_file = os.path.join(session_dir, 'predicted_csv', f'{impact_id}_ball.csv')
            assert os.path.exists(pred_csv_file), f'{pred_csv_file} does not exist.'
            pred_df = pd.read_csv(pred_csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)

            f_file = np.array([os.path.join(impact_dir, f'{f_id}.{IMG_FORMAT}') for f_id in pred_df['Frame']])
            x, y, v = np.array(pred_df['X_GT']), np.array(pred_df['Y_GT']), np.array(pred_df['Visibility_GT'])
            x_pred, y_pred, v_pred = np.array(pred_df['X']), np.array(pred_df['Y']), np.array(pred_df['Visibility'])
            inpaint = np.array(pred_df['Inpaint_Mask'])

            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            # Sliding on the frame sequence
            last_idx = -1
            for i in range(0, len(f_file), self.sliding_step):
                tmp_idx, tmp_coor, tmp_coor_pred, tmp_vis, tmp_vis_pred, tmp_inpaint  = [], [], [], [], [], []
                # Construct a single input sequence
                for f in range(self.seq_len):
                    if i+f < len(f_file):
                        tmp_idx.append((impact_i, i+f))
                        tmp_coor.append((x[i+f], y[i+f]))
                        tmp_coor_pred.append((x_pred[i+f], y_pred[i+f]))
                        tmp_vis.append(v[i+f])
                        tmp_vis_pred.append(v_pred[i+f])
                        tmp_inpaint.append(inpaint[i+f])
                    else:
                        # Padding the last sequence if imcompleted
                        if self.padding:
                            tmp_idx.append((impact_i, last_idx))
                            tmp_coor.append((x[last_idx], y[last_idx]))
                            tmp_coor_pred.append((x_pred[last_idx], y_pred[last_idx]))
                            tmp_vis.append(v[last_idx])
                            tmp_vis_pred.append(v_pred[last_idx])
                            tmp_inpaint.append(inpaint[last_idx])
                        else:
                            break

                # Append the input sequence
                if len(tmp_idx) == self.seq_len:
                    assert len(tmp_idx) == len(tmp_coor) == len(tmp_coor_pred) == \
                           len(tmp_vis) == len(tmp_vis_pred) == len(tmp_inpaint), \
                            f'Length of frames, coordinates, predicted coordinates,\
                            visibilities, predicted visibilities and inpaint masks are not equal.'
                    id = np.concatenate((id, [tmp_idx]), axis=0)
                    coor = np.concatenate((coor, [tmp_coor]), axis=0)
                    coor_pred = np.concatenate((coor_pred, [tmp_coor_pred]), axis=0)
                    vis = np.concatenate((vis, [tmp_vis]), axis=0)
                    pred_vis = np.concatenate((pred_vis, [tmp_vis_pred]), axis=0)
                    inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)
            
            return dict(id=id, coor=coor, coor_pred=coor_pred, vis=vis, pred_vis=pred_vis, inpaint_mask=inpaint_mask)
        
    def _gen_input_from_pred_dict(self) -> tuple[dict, dict]:
        """ Generate input sequences from a prediction dictionary. """
        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
        pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)
       
        # these are assigned to be not None before calling this function
        x_pred, y_pred, vis_pred = self.pred_dict['X'], self.pred_dict['Y'], self.pred_dict['Visibility'] #type:ignore
        inpaint = self.pred_dict['Inpaint_Mask'] # type: ignore 
        
        assert len(x_pred) == len(y_pred) == len(vis_pred) == len(inpaint), \
            f'Length of x_pred, y_pred, vis_pred and inpaint are not equal.'
        
        # Sliding on the frame sequence
        last_idx = -1
        for i in range(0, len(inpaint), self.sliding_step):
            tmp_idx, tmp_coor_pred, tmp_vis_pred, tmp_inpaint = [], [], [], []
            # Construct a single input sequence
            for f in range(self.seq_len):
                if i+f < len(inpaint):
                    tmp_idx.append((0, i+f))
                    tmp_coor_pred.append((x_pred[i+f], y_pred[i+f]))
                    tmp_vis_pred.append(vis_pred[i+f])
                    tmp_inpaint.append(inpaint[i+f])
                    last_idx = i+f
                else:
                    # Padding the last sequence if imcompleted
                    if self.padding:
                        tmp_idx.append((0, last_idx))
                        tmp_coor_pred.append((x_pred[last_idx], y_pred[last_idx]))
                        tmp_vis_pred.append(vis_pred[last_idx])
                        tmp_inpaint.append(inpaint[last_idx])
                    else:
                        break
                
            if len(tmp_idx) == self.seq_len:
                assert len(tmp_coor_pred) == len(tmp_inpaint), \
                    f'Length of predicted coordinates and inpaint masks are not equal.'
                id = np.concatenate((id, [tmp_idx]), axis=0)
                coor_pred = np.concatenate((coor_pred, [tmp_coor_pred]), axis=0)
                pred_vis = np.concatenate((pred_vis, [tmp_vis_pred]), axis=0)
                inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)
        
        return dict(id=id, coor_pred=coor_pred, pred_vis=pred_vis, inpaint_mask=inpaint_mask),\
               dict(img_scaler=self.pred_dict['Img_scaler'], img_shape=self.pred_dict['Img_shape']) # type: ignore
    
    def _get_heatmap(
        self,
        cx,
        cy,
        heatmap_type: str = "circle",
        sigma_x: float | None = None,
        sigma_y: float | None = None,
        angle_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Generate a heatmap centered at (cx, cy).

        Supported types:
            - "circle": binary circular disk
            - "ellipse": binary rotated ellipse
            - "gaussian": isotropic Gaussian
            - "gaussian_ellipse": anisotropic rotated Gaussian

        Args:
            cx, cy: center coordinates in resized image space
            heatmap_type: type of heatmap to generate
            sigma_x: horizontal radius / std
            sigma_y: vertical radius / std
            angle_deg: rotation angle in degrees, used for ellipse / gaussian_ellipse
        """
        if cx == cy == 0:
            return np.zeros((1, self.HEIGHT, self.WIDTH), dtype=np.float32)

        sigma_x = float(sigma_x if sigma_x is not None else self.sigma)
        sigma_y = float(sigma_y if sigma_y is not None else self.sigma)

        # Grid in pixel coordinates
        x, y = np.meshgrid(
            np.arange(self.WIDTH, dtype=np.float32),
            np.arange(self.HEIGHT, dtype=np.float32),
        )

        dx = x - float(cx)
        dy = y - float(cy)

        if heatmap_type == "circle":
            dist2 = dx**2 + dy**2
            heatmap = (dist2 <= sigma_x**2).astype(np.float32) * self.mag

        elif heatmap_type == "ellipse":
            theta = np.deg2rad(angle_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # Rotate coordinates into ellipse frame
            xr = cos_t * dx + sin_t * dy
            yr = -sin_t * dx + cos_t * dy

            ellipse_eq = (xr**2) / (sigma_x**2 + 1e-6) + (yr**2) / (sigma_y**2 + 1e-6)
            heatmap = (ellipse_eq <= 1.0).astype(np.float32) * self.mag

        elif heatmap_type == "gaussian":
            dist2 = dx**2 + dy**2
            heatmap = np.exp(-dist2 / (2.0 * sigma_x**2 + 1e-6)).astype(np.float32)
            heatmap *= self.mag

        elif heatmap_type == "gaussian_ellipse":
            theta = np.deg2rad(angle_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # Rotate coordinates into Gaussian principal axes
            xr = cos_t * dx + sin_t * dy
            yr = -sin_t * dx + cos_t * dy

            heatmap = np.exp(
                -0.5 * (
                    (xr**2) / (sigma_x**2 + 1e-6) +
                    (yr**2) / (sigma_y**2 + 1e-6)
                )
            ).astype(np.float32)
            heatmap *= self.mag

        else:
            raise ValueError(
                f"Unsupported heatmap_type: {heatmap_type}. "
                f"Choose from ['circle', 'ellipse', 'gaussian', 'gaussian_ellipse']"
            )

        return heatmap.reshape(1, self.HEIGHT, self.WIDTH)

    def __len__(self):
        """ Return the number of data in the dataset. """
        return len(self.data_dict['id'])

    def __getitem__(self, idx):
        """ Return the data of the given index.

            For training and evaluation:
                'heatmap': Return data_idx, frames, heatmaps, tmp_coor, tmp_vis
                'coordinate': Return data_idx, coor_pred, inpaint

            For inference:
                'heatmap': Return data_idx, frames
                'coordinate': Return data_idx, coor_pred, inpaint
        """
        if self.frame_arr is not None:
            data_idx = self.data_dict['id'][idx] # (L,)
            imgs = self.frame_arr[data_idx[:, 1], ...] # (L, H, W, 3)

            if self.bg_mode:
                median_img = self.median
            
            # Process the frame sequence
            frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
            for i in range(self.seq_len):
                img = Image.fromarray(imgs[i])
                if self.bg_mode == 'subtract':
                    img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8')) # type: ignore
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = img.reshape(1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    diff_img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8')) # type: ignore
                    diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = np.moveaxis(img, -1, 0)
                    img = np.concatenate((img, diff_img), axis=0)
                else:
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = np.moveaxis(img, -1, 0)
                
                frames = np.concatenate((frames, img), axis=0)
            
            if self.bg_mode == 'concat':
                frames = np.concatenate((median_img, frames), axis=0) # type: ignore
            
            # Normalization
            frames /= 255.

            return data_idx, frames

        elif self.pred_dict is not None:
            data_idx = self.data_dict['id'][idx] # (L,)
            coor_pred = self.data_dict['coor_pred'][idx] # (L, 2)
            inpaint = self.data_dict['inpaint_mask'][idx].reshape(-1, 1) # (L, 1)
            w, h = self.img_config['img_shape']
            
            # Normalization
            coor_pred[:, 0] = coor_pred[:, 0] / w
            coor_pred[:, 1] = coor_pred[:, 1] / h

            return data_idx, coor_pred, inpaint

        elif self.data_mode == 'heatmap':
            if self.frame_alpha > 0:
                data_idx = self.data_dict['id'][idx] # (L,)
                frame_file = self.data_dict['frame_file'][idx] # (L,)
                coor = self.data_dict['coor'][idx] # (L, 2)
                vis = self.data_dict['vis'][idx] # (L,)
                w, h = self.img_config['img_shape'][data_idx[0][0]]
                w_scaler, h_scaler = self.img_config['img_scaler'][data_idx[0][0]]
                heatmap_type, sigma_x, sigma_y, angle_deg = self.img_config['heatmap_config'][data_idx[0][0]]

                if self.bg_mode:
                    file_format_str = os.path.join('{}', 'frame', '{}','{}.'+IMG_FORMAT)
                    session_dir, rally_id, _ = parse.parse(file_format_str, frame_file[0])#'{}/frame/{}/{}.png', frame_file[0])
                    median_file = os.path.join(session_dir, 'median.npz') if os.path.exists(os.path.join(session_dir, 'median.npz')) else os.path.join(session_dir, 'frame', rally_id, 'median.npz')
                    assert os.path.exists(median_file), f'{median_file} does not exist.'
                    median_img = np.load(median_file)['median']
                
                # Frame mixup
                # Sample the mixing ratio
                lamb = np.random.beta(self.frame_alpha, self.frame_alpha)

                # Initialize the previous frame data
                prev_img = Image.open(frame_file[0])
                if self.bg_mode == 'subtract':
                    prev_img = Image.fromarray(np.sum(np.absolute(prev_img - median_img), 2).astype('uint8')) # type: ignore
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = prev_img.reshape(1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    diff_img = Image.fromarray(np.sum(np.absolute(prev_img - median_img), 2).astype('uint8')) # type: ignore
                    diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = np.moveaxis(prev_img, -1, 0)
                    prev_img = np.concatenate((prev_img, diff_img), axis=0)
                else:
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = np.moveaxis(prev_img, -1, 0)

                prev_coor = coor[0]
                prev_vis = vis[0]
                prev_heatmap = self._get_heatmap(cx=int(coor[0][0]/ w_scaler), cy=int(coor[0][1]/ h_scaler), heatmap_type=heatmap_type, sigma_x=sigma_x, sigma_y=sigma_y, angle_deg=angle_deg)
                
                # Keep first dimension as timestamp for resample
                if self.bg_mode == 'subtract':
                    frames = prev_img.reshape(1, 1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    frames = prev_img.reshape(1, 4, self.HEIGHT, self.WIDTH)
                else:
                    frames = prev_img.reshape(1, 3, self.HEIGHT, self.WIDTH)

                tmp_coor = prev_coor.reshape(1, -1)
                tmp_vis = prev_vis.reshape(1, -1)
                heatmaps = prev_heatmap
                
                # Read image and generate heatmap
                for i in range(1, self.seq_len):
                    cur_img = Image.open(frame_file[i])
                    if self.bg_mode == 'subtract':
                        cur_img = Image.fromarray(np.sum(np.absolute(cur_img - median_img), 2).astype('uint8')) # type: ignore
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = cur_img.reshape(1, self.HEIGHT, self.WIDTH)
                    elif self.bg_mode == 'subtract_concat':
                        diff_img = Image.fromarray(np.sum(np.absolute(cur_img - median_img), 2).astype('uint8')) # type: ignore
                        diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = np.moveaxis(cur_img, -1, 0)
                        cur_img = np.concatenate((cur_img, diff_img), axis=0)
                    else:
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = np.moveaxis(cur_img, -1, 0)

                    inter_img = prev_img * lamb + cur_img * (1 - lamb)

                    # Linear interpolation
                    if vis[i] == 0:
                        inter_coor = prev_coor
                        inter_vis = prev_vis
                        cur_heatmap = prev_heatmap
                        inter_heatmap = cur_heatmap
                    elif prev_vis == 0 or math.sqrt(pow(prev_coor[0]-coor[i][0], 2)+pow(prev_coor[1]-coor[i][1], 2)) < 10:
                        inter_coor = coor[i]
                        inter_vis = vis[i]
                        cur_heatmap = self._get_heatmap(cx=int(inter_coor[0]/ w_scaler), cy=int(inter_coor[1]/ h_scaler), heatmap_type=heatmap_type, sigma_x=sigma_x, sigma_y=sigma_y, angle_deg=angle_deg)
                        inter_heatmap = cur_heatmap
                    else:
                        inter_coor = coor[i]
                        inter_vis = vis[i]
                        cur_heatmap = self._get_heatmap(cx=int(coor[i][0]/ w_scaler), cy=int(coor[i][1]/ h_scaler), heatmap_type=heatmap_type, sigma_x=sigma_x, sigma_y=sigma_y, angle_deg=angle_deg)
                        inter_heatmap = prev_heatmap * lamb + cur_heatmap * (1 - lamb)
                    
                    tmp_coor = np.concatenate((tmp_coor, inter_coor.reshape(1, -1), coor[i].reshape(1, -1)), axis=0)
                    tmp_vis = np.concatenate((tmp_vis, np.array([inter_vis]).reshape(1, -1), np.array([vis[i]]).reshape(1, -1)), axis=0)
                    frames = np.concatenate((frames, inter_img[None,:,:,:], cur_img[None,:,:,:]), axis=0)
                    heatmaps = np.concatenate((heatmaps, inter_heatmap, cur_heatmap), axis=0)
                    
                    prev_img, prev_heatmap, prev_coor, prev_vis = cur_img, cur_heatmap, coor[i], vis[i]
                
                # Resample input sequence
                rand_id = np.random.choice(len(frames), self.seq_len, replace=False)
                rand_id = np.sort(rand_id)
                tmp_coor = tmp_coor[rand_id]
                tmp_vis = tmp_vis[rand_id]
                frames = frames[rand_id]
                heatmaps = heatmaps[rand_id]
                
                if self.bg_mode == 'concat':
                    median_img = Image.fromarray(median_img.astype('uint8')) # type: ignore
                    median_img = np.array(median_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    median_img = np.moveaxis(median_img, -1, 0)
                    frames = np.concatenate((median_img.reshape(1, 3, self.HEIGHT, self.WIDTH), frames), axis=0)
                
                # Reshape to model input format
                frames = frames.reshape(-1, self.HEIGHT, self.WIDTH)

                # Normalization
                frames /= 255.
                tmp_coor[:, 0] = tmp_coor[:, 0] / w
                tmp_coor[:, 1] = tmp_coor[:, 1] / h

                return data_idx, frames, heatmaps, tmp_coor, tmp_vis
            else:
                data_idx = self.data_dict['id'][idx]
                frame_file = self.data_dict['frame_file'][idx]
                coor = self.data_dict['coor'][idx]
                vis = self.data_dict['vis'][idx]
                w, h = self.img_config['img_shape'][data_idx[0][0]]
                w_scaler, h_scaler = self.img_config['img_scaler'][data_idx[0][0]]

                # Read median image
                if self.bg_mode:
                    file_format_str = os.path.join('{}', 'frame', '{}','{}.'+IMG_FORMAT)
                    session_dir, rally_id, _ = parse.parse(file_format_str, frame_file[0])#'{}/frame/{}/{}.png', frame_file[0])
                    median_file = os.path.join(session_dir, 'median.npz') if os.path.exists(os.path.join(session_dir, 'median.npz')) else os.path.join(session_dir, 'frame', rally_id, 'median.npz')
                    assert os.path.exists(median_file), f'{median_file} does not exist.'
                    median_img = np.load(median_file)['median']

                frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
                heatmaps = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
                
                # Read image and generate heatmap
                for i in range(self.seq_len):
                    img = Image.open(frame_file[i])
                    if self.bg_mode == 'subtract':
                        img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8')) # type: ignore
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = img.reshape(1, self.HEIGHT, self.WIDTH)
                    elif self.bg_mode == 'subtract_concat':
                        diff_img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8')) # type: ignore
                        diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = np.moveaxis(img, -1, 0)
                        img = np.concatenate((img, diff_img), axis=0)
                    else:
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = np.moveaxis(img, -1, 0)
                    
                    heatmap = self._get_heatmap(cx=int(coor[i][0]/w_scaler), cy=int(coor[i][1]/h_scaler), heatmap_type=self.img_config['heatmap_config'][data_idx[i][0]][0], sigma_x=self.img_config['heatmap_config'][data_idx[i][0]][1], sigma_y=self.img_config['heatmap_config'][data_idx[i][0]][2], angle_deg=self.img_config['heatmap_config'][data_idx[i][0]][3])
                    frames = np.concatenate((frames, img), axis=0)
                    heatmaps = np.concatenate((heatmaps, heatmap), axis=0)
                
                if self.bg_mode == 'concat':
                    median_img = Image.fromarray(median_img.astype('uint8')) # type: ignore
                    median_img = np.array(median_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    median_img = np.moveaxis(median_img, -1, 0)
                    frames = np.concatenate((median_img, frames), axis=0)

                # Normalization
                frames /= 255.
                coor[:, 0] = coor[:, 0] / w
                coor[:, 1] = coor[:, 1] / h

                return data_idx, frames, heatmaps, coor, vis
        
        elif self.data_mode == 'coordinate':
            data_idx = self.data_dict['id'][idx] # (L,)
            coor = self.data_dict['coor'][idx] # (L, 2)
            coor_pred = self.data_dict['coor_pred'][idx] # (L, 2)
            vis = self.data_dict['vis'][idx] # (L,)
            vis_pred = self.data_dict['pred_vis'][idx] # (L,)
            inpaint = self.data_dict['inpaint_mask'][idx] # (L,)
            w, h = self.img_config['img_shape'][data_idx[0][0]]
            
            # Normalization
            coor[:, 0] = coor[:, 0] / self.WIDTH
            coor[:, 1] = coor[:, 1] / self.HEIGHT
            coor_pred[:, 0] = coor_pred[:, 0] / self.WIDTH
            coor_pred[:, 1] = coor_pred[:, 1] / self.HEIGHT

            return data_idx, coor_pred, coor, vis_pred.reshape(-1, 1), vis.reshape(-1, 1), inpaint.reshape(-1, 1)
        else:
            raise NotImplementedError