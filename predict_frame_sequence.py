import os
import re
import argparse
from collections import deque

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import Shuttlecock_Trajectory_Dataset
from predict import predict
from test import get_ensemble_weight, generate_inpaint_mask
from utils.general import COOR_TH, HEIGHT, WIDTH, draw_traj, get_model, write_pred_csv, write_pred_video


SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def natural_sort_key(path):
    """Sort file names like 1.png, 2.png, 10.png."""
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def get_frame_files(frame_dir):
    frame_files = []
    for name in os.listdir(frame_dir):
        file_path = os.path.join(frame_dir, name)
        if os.path.isfile(file_path) and os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
            frame_files.append(file_path)

    frame_files = sorted(frame_files, key=natural_sort_key)
    if not frame_files:
        raise ValueError(f'No image files found in {frame_dir}')

    return frame_files


def load_frames(frame_files):
    frame_list = []
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is None:
            raise ValueError(f'Failed to read image: {frame_file}')
        frame_list.append(frame)

    first_shape = frame_list[0].shape[:2]
    for frame_file, frame in zip(frame_files, frame_list):
        if frame.shape[:2] != first_shape:
            raise ValueError(f'Image size mismatch: {frame_file} has shape {frame.shape[:2]}, expected {first_shape}')

    return frame_list


def write_video_from_frames(frame_list, save_file, fps):
    h, w = frame_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_file, fourcc, fps, (w, h))

    for frame in frame_list:
        writer.write(frame)

    writer.release()


def save_overlay_frames(frame_list, pred_dict, save_dir, traj_len=8):
    os.makedirs(save_dir, exist_ok=True)
    pred_queue = deque()
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']

    for idx, frame in enumerate(frame_list):
        if len(pred_queue) >= traj_len:
            pred_queue.pop()

        pred_queue.appendleft([x_pred[idx], y_pred[idx]]) if vis_pred[idx] else pred_queue.appendleft(None)
        overlay = draw_traj(frame.copy(), pred_queue, color='yellow')
        cv2.imwrite(os.path.join(save_dir, f'{idx:06d}.png'), overlay)


def run_tracknet_inference(tracknet, frame_list, batch_size, eval_mode, bg_mode, seq_len, device):
    num_workers = batch_size if batch_size <= 16 else 16
    h, w = frame_list[0].shape[:2]
    img_scaler = (w / WIDTH, h / HEIGHT)
    frame_arr = np.array(frame_list)[:, :, :, ::-1]

    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
                 'Img_scaler': img_scaler, 'Img_shape': (w, h)}

    tracknet.eval()

    if eval_mode == 'nonoverlap':
        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=seq_len,
            sliding_step=seq_len,
            data_mode='heatmap',
            bg_mode=bg_mode,
            frame_arr=frame_arr,
            padding=True,
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        for i, x in tqdm(data_loader):
            x = x.float().to(device)
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()

            tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                pred_dict[key].extend(tmp_pred[key])
    else:
        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=seq_len,
            sliding_step=1,
            data_mode='heatmap',
            bg_mode=bg_mode,
            frame_arr=frame_arr,
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        video_len = len(frame_list)

        num_sample, sample_count = video_len - seq_len + 1, 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len)
        frame_i = torch.arange(seq_len - 1, -1, -1)
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
        weight = get_ensemble_weight(seq_len, eval_mode)

        for i, x in tqdm(data_loader):
            x = x.float().to(device)
            b_size = i.shape[0]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()

            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    y_pred = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    y_pred = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)

                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        y_pred = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

            tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                pred_dict[key].extend(tmp_pred[key])

            y_pred_buffer = y_pred_buffer[-buffer_size:]

    return pred_dict


def run_inpaintnet_inference(inpaintnet, tracknet_pred_dict, batch_size, eval_mode, seq_len, device):
    num_workers = batch_size if batch_size <= 16 else 16
    h = tracknet_pred_dict['Img_shape'][1]
    tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h * 0.05)
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

    inpaintnet.eval()

    if eval_mode == 'nonoverlap':
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate',
                                                 pred_dict=tracknet_pred_dict, padding=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        for i, coor_pred, inpaint_mask in tqdm(data_loader):
            coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
            with torch.no_grad():
                coor_inpaint = inpaintnet(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
                coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

            th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
            coor_inpaint[th_mask] = 0.

            tmp_pred = predict(i, c_pred=coor_inpaint, img_scaler=tracknet_pred_dict['Img_scaler'])
            for key in tmp_pred.keys():
                pred_dict[key].extend(tmp_pred[key])
    else:
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate',
                                                 pred_dict=tracknet_pred_dict)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        weight = get_ensemble_weight(seq_len, eval_mode)

        num_sample, sample_count = len(dataset), 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len)
        frame_i = torch.arange(seq_len - 1, -1, -1)
        coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)

        for i, coor_pred, inpaint_mask in tqdm(data_loader):
            coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
            b_size = i.shape[0]
            with torch.no_grad():
                coor_inpaint = inpaintnet(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
                coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

            th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
            coor_inpaint[th_mask] = 0.

            coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    coor_inpaint = coor_inpaint_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    coor_inpaint = (coor_inpaint_buffer[batch_i + b, frame_i] * weight[:, None]).sum(0)

                ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                    coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        coor_inpaint = coor_inpaint_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                        ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

            th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
            ensemble_coor_inpaint[th_mask] = 0.

            tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=tracknet_pred_dict['Img_scaler'])
            for key in tmp_pred.keys():
                pred_dict[key].extend(tmp_pred[key])

            coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

    return pred_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_dir', type=str, required=True, help='directory containing the ordered image sequence')
    parser.add_argument('--tracknet_file', type=str, required=True, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--save_dir', type=str, default='pred_frame_sequence', help='directory to save outputs')
    parser.add_argument('--fps', type=int, default=30, help='fps to use when creating videos from the frame sequence')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on the overlay')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='inference device')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but torch.cuda.is_available() is False')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    os.makedirs(args.save_dir, exist_ok=True)
    sequence_name = os.path.basename(os.path.normpath(args.frame_dir))
    input_video_file = os.path.join(args.save_dir, f'{sequence_name}_input.mp4')
    overlay_video_file = os.path.join(args.save_dir, f'{sequence_name}_overlay.mp4')
    csv_file = os.path.join(args.save_dir, f'{sequence_name}_ball.csv')
    overlay_frame_dir = os.path.join(args.save_dir, f'{sequence_name}_overlay_frames')

    print('Loading frames...')
    frame_files = get_frame_files(args.frame_dir)
    frame_list = load_frames(frame_files)

    print('Writing input video...')
    write_video_from_frames(frame_list, input_video_file, args.fps)

    print('Loading model checkpoints...')
    tracknet_ckpt = torch.load(args.tracknet_file, map_location=device)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
    tracknet.load_state_dict(tracknet_ckpt['model'])

    if args.eval_mode != 'nonoverlap' and len(frame_list) < tracknet_seq_len:
        raise ValueError(f'Need at least {tracknet_seq_len} frames for eval_mode={args.eval_mode}, got {len(frame_list)}')

    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file, map_location=device)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').to(device)
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
        if args.eval_mode != 'nonoverlap' and len(frame_list) < inpaintnet_seq_len:
            raise ValueError(f'Need at least {inpaintnet_seq_len} frames for InpaintNet with eval_mode={args.eval_mode}, got {len(frame_list)}')
    else:
        inpaintnet = None

    print(f'Running inference on {device}...')
    tracknet_pred_dict = run_tracknet_inference(tracknet, frame_list, args.batch_size, args.eval_mode, bg_mode, tracknet_seq_len, device)

    if inpaintnet is not None:
        print('Running InpaintNet refinement...')
        pred_dict = run_inpaintnet_inference(inpaintnet, tracknet_pred_dict, args.batch_size, args.eval_mode, inpaintnet_seq_len, device)
    else:
        pred_dict = tracknet_pred_dict

    print('Writing prediction outputs...')
    write_pred_csv(pred_dict, save_file=csv_file)
    write_pred_video(input_video_file, pred_dict, save_file=overlay_video_file, traj_len=args.traj_len)
    save_overlay_frames(frame_list, pred_dict, save_dir=overlay_frame_dir, traj_len=args.traj_len)

    print(f'Input video: {input_video_file}')
    print(f'Prediction csv: {csv_file}')
    print(f'Overlay video: {overlay_video_file}')
    print(f'Overlay frames: {overlay_frame_dir}')
