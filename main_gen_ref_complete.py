%%writefile /tmp/script.py
import os
import argparse
import glob
from torch.cuda import is_available

#import functions from local files
import sys
sys.path.append('/kaggle/input/datasets/tausifr/pythonlib-folder/')
sys.path.append('/kaggle/input/datasets/tausifr/grip-folder3/')

# --- config.py code ---
import yaml
import numpy as np
from time import time
from tqdm import tqdm

import torch
from scipy.optimize import linear_sum_assignment

DEFAULT_OPT = {
    'resources_path': None,
    'fps': 25,
    'read_stride': 96,
    'rec_stride': 32,
    'det_stride': 12,
    'face_det': {
        'size_threshold': 75,
        'score_threshold': 0.7,
        'iou_threshold': 0.4,
    },
    'audio': {
        'sampling_rate': 16000,
        'norm_target_dBFS': -30,
        'num_fft': 512,
        'window_step': 10,
        'window_length': 25,
    },
    'model': 'poiforensics',
    'percentile': 5,
    'final_mean': 7,
    'dist_normalization': None,
    'output_ffmpeg_params': {
        '-c:v': 'libx264', '-profile:v': 'high', '-level:v': '4.0',
        '-pix_fmt': 'yuv420p', '-crf': '35',
    },
}

def create_opt(**opt):
    opt = {key: opt.get(key, DEFAULT_OPT[key]) for key in DEFAULT_OPT}
    if not isinstance(opt['model'], dict):
        if opt['model'] == 'poiforensics':
            # Directly use the contents of poiforensics.yaml
            opt['model'] = {
                'type': 'poi_forensics',
                'architecture_audio': 'enc_resnet50',
                'architecture_video': 'enc_resnet50',
                'factor_len': 1,
                'feats_len': 256,
                'face_size': 224,
                'clip_length': 75,
                'clip_video_stride': 3,
                'clip_stride': 25,
                'clip_ref_stride': 25,
                'weights': './weights_audiovideo.th',
            }
        else:
            file_model = './config/%s.yaml' % opt['model']
            with open(file_model) as fid:
                opt['model'] = yaml.load(fid, Loader=yaml.FullLoader)['model']
    if opt['dist_normalization'] is None:
        opt['dist_normalization'] = opt['model']['type']=='poi_forensics'
    return opt

def get_extraction_opt(opt):
    return {key: opt[key] for key in [
        'fps',
        'audio',
        'model',
    ]}


# --- util_write.py code ---
import cv2

from skvideo.io import FFmpegWriter
from scipy.special import expit as sigmoid


def drawBox(frame_np, box, color_box=(0, 0, 255)):
    return cv2.rectangle(frame_np, (int(box[0]), int(box[1])), (int(box[2]) + 1, int(box[3]) + 1), color_box, 10)


def drawText(frame_np, box, txt, color_txt=(255, 255, 255)):
    if int(box[1]) < 50:
        frame_np = cv2.putText(frame_np, txt, (int(box[0]) + 5, int(box[1]) + (int(box[3]) - int(box[1]))//4),
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_txt, 5)
    else:
        frame_np = cv2.putText(frame_np, txt, (int(box[0]) + 3, int(box[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_txt, 5)

    return frame_np


def drawPoints(frame_np, points, color=(255, 255, 255)):
    for p in points:
        frame_np = cv2.drawMarker(frame_np, (p[0], p[1]), color,  cv2.MARKER_TILTED_CROSS, thickness=5)
    return frame_np


def drawHeat(frame_np, box, x):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    x = np.float32(x)
    img = frame_np[y1:y2, x1:x2]
    x = x + np.mean(img, -1, keepdims=True) - np.mean(x, -1, keepdims=True)
    frame_np[y1:y2, x1:x2] = np.uint8(x.clip(0, 255))

    return frame_np


class WritingVideo:
    def __init__(self, filename, fps, tag_frame='frames_out_bgr',
                 vid_configure={'-c:v': 'libx264', '-preset': 'ultrafast', '-crf': '35'}):
        self.filename = filename
        self.fps = fps
        self.tag_frame = tag_frame
        self.vid_configure = vid_configure

        self.count_frame = 0
        self.video_out = None

        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

    def __enter__(self):
        self.count_frame = 0
        self.video_out = FFmpegWriter(self.filename, inputdict={'-r': str(self.fps)},
                                      outputdict=self.vid_configure, verbosity=0)
        return self

    def __call__(self, inp):
        assert self.tag_frame in inp
        if len(self.tag_frame)==0:
            return inp

        for index_f in np.argsort(inp['frames_inds']):
            index = inp['frames_inds'][index_f]
            frame = cv2.cvtColor(inp[self.tag_frame][index_f], cv2.COLOR_BGR2RGB)
            while self.count_frame <= index:
                self.video_out.writeFrame(frame)
                self.count_frame = self.count_frame + 1

        return inp

    def __exit__(self, type, value, tb):
        self.count_frame = 0
        try:
            self.video_out.close()
        except:
            pass
        self.video_out = None


class GenFrameBoxes:
    def __init__(self, tag_boxes='boxes', return_frame=False):
        self.tag_boxes = tag_boxes
        self.return_frame = return_frame
        self.color_loop = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

    def reset(self):
        return self

    def __call__(self, inp):
        assert 'frames_bgr' in inp
        imgs = inp['frames_bgr']
        ids = inp['frames_inds']
        if 'image_inds' in inp:
            image_inds = inp['image_inds']
            boxes = inp[self.tag_boxes]
            tracks = inp['id_track']
        else:
            image_inds = list()
            boxes = list()
            tracks = list()
        if not self.return_frame:
            del inp['frames_bgr']

        list_frames_out_bgr = list()
        for index_f, frame in zip(ids, imgs):
            frame = np.copy(frame)
            for index in range(len(image_inds)):
                if image_inds[index] == index_f:
                    frame = drawBox(frame, boxes[index], self.color_loop[tracks[index] % len(self.color_loop)])
                    frame = drawText(frame, boxes[index], '%d' % tracks[index], color_txt=(255, 255, 255))

            list_frames_out_bgr.append(frame)

        inp['frames_out_bgr'] = list_frames_out_bgr
        return inp


class GenFrameHeat:
    def __init__(self, cmap, fit_lim=sigmoid, tag_boxes='face_boxes', return_frame=False):
        self.tag_boxes = tag_boxes
        self.return_frame = return_frame
        self.cmap_bgr = 255*cmap[:, ::-1]
        self.fit_lim = fit_lim

    def reset(self):
        return self

    def fit_size(self, box, x):
        h = int(box[3]) - int(box[1])
        w = int(box[2]) - int(box[0])
        if h > w:
            wm = int(round(h * x.shape[1] / x.shape[0]))
            x = cv2.resize(x, (wm, h), interpolation=cv2.INTER_LINEAR)
        else:
            hm = int(round(w * x.shape[0] / x.shape[1]))
            x = cv2.resize(x, (w, hm), interpolation=cv2.INTER_LINEAR)

        s_w = (x.shape[1] - w) // 2
        s_h = (x.shape[0] - h) // 2
        x = x[s_h:(s_h+h), s_w:(s_w+w)]
        return x

    def apply_cmap(self, x_bgr):
        x_bgr = np.int64(np.round(x_bgr * len(self.cmap_bgr)).clip(0, len(self.cmap_bgr)-1))
        return self.cmap_bgr[x_bgr, :]

    def __call__(self, inp):
        assert 'frames_bgr' in inp
        imgs = inp['frames_bgr']
        ids = inp['frames_inds']
        if 'image_inds' in inp:
            image_inds = inp['image_inds']
            boxes = inp[self.tag_boxes]
            mappreds = inp['mappreds']
        else:
            image_inds = list()
            boxes = list()
            mappreds = list()
        if not self.return_frame:
            del inp['frames_bgr']

        list_frames_out_bgr = list()
        for index_f, frame in zip(ids, imgs):
            frame = np.copy(frame)
            for index in range(len(image_inds)):
                if image_inds[index] == index_f:
                    x = self.apply_cmap(self.fit_lim(self.fit_size(boxes[index], mappreds[index])))
                    frame = drawHeat(frame, boxes[index], x)

            list_frames_out_bgr.append(frame)

        inp['frames_out_bgr'] = list_frames_out_bgr
        return inp


class OutputBoxes:
    def __init__(self, outputfile, margin=25, return_frame=False, color=(180, 180, 180)):
        dat = np.load(outputfile)
        self.boxes = dat['embs_boxes']
        self.ranges = np.asarray([[_[0] + margin, _[1] - margin] for _ in dat['embs_range'].astype(np.int64)])
        if 'embs_points' in dat:
            self.points = np.asarray([_[margin: len(_)-margin] for _ in dat['embs_points']])
        else:
            self.points = None
        self.dists = dat['embs_dists']
        self.id_track = dat['embs_track']
        while len(self.dists.shape) > 1:
            self.dists = self.dists[..., -1]
        del dat
        self.return_frame = return_frame
        self.color = color

    def reset(self):
        return self

    def __call__(self, inp):
        assert 'frames_bgr' in inp
        imgs = inp['frames_bgr']
        ids = inp['frames_inds']
        if not self.return_frame:
            del inp['frames_bgr']

        list_frames_out_bgr = list()
        for index_f, frame in zip(ids, imgs):
            frame = np.copy(frame)

            sel = (self.ranges[:, 0] <= index_f) & (index_f < self.ranges[:, 1])
            boxf = self.boxes[sel]
            tracks = self.id_track[sel]
            distsf = self.dists[sel]
            rp = self.ranges[sel]
            if self.points is not None:
                pp = self.points[sel]
            else:
                pp = None

            for index in range(len(boxf)):
                if np.isnan(boxf[index][0]):
                    frame = drawText(frame, (0, 50, 50, 100), '%.3f' % distsf[index], color_txt=self.color)
                else:
                    frame = drawBox(frame, boxf[index], self.color)
                    frame = drawText(frame, boxf[index], '%.3f' % distsf[index], color_txt=self.color)
                if pp is not None:
                    try:
                        xy = pp[index][index_f-rp[index][0]].reshape(-1, 2)
                        frame = drawPoints(frame, xy, self.color)
                    except:
                        pass


            list_frames_out_bgr.append(frame)

        inp['frames_out_bgr'] = list_frames_out_bgr
        return inp


class WritingClips:
    def __init__(self, filedir, write_one=False):
        self.list_track = dict()
        self.write_one = write_one
        self.filedir = filedir
        os.makedirs(self.filedir, exist_ok=True)

    def __enter__(self):
        self.list_track = dict()
        return self

    def __call__(self, inp):
        assert 'boxes' in inp
        assert 'image_inds' in inp
        assert 'id_track' in inp
        assert 'points' in inp
        assert 'face_bgr' in inp
        assert 'face_start' in inp
        flag_landmarks = 'landmarks68' in inp

        for index_f in np.argsort(inp['image_inds']):
            index = inp['image_inds'][index_f]
            track = inp['id_track'][index_f]
            start = inp['face_start'][index_f]
            point = [[inp['points'][index_f][2 * j] - start[0],
                      inp['points'][index_f][2 * j + 1] - start[1]] for j in range(5)]
            if flag_landmarks:
                lm68 = np.asarray(inp['landmarks68'][index_f]) - [start, ]
            else:
                lm68 = None
            box = inp['boxes'][index_f]
            box = [box[0]-start[0], box[1]-start[1],
                   box[2]-start[0], box[3]-start[1]]

            if track not in self.list_track:
                self.list_track[track] = index

            count = index - self.list_track[track]
            if self.write_one:
                if count == 0:
                    filepng = os.path.join(self.filedir, 'track_%d.png' % track)
                    cv2.imwrite(filepng, inp['face_bgr'][index_f])
                continue

            filepng = os.path.join(self.filedir, 'crop_%d_%d.png' % (track, count))
            filenpz = os.path.join(self.filedir, 'crop_%d_%d.png.npz' % (track, count))

            cv2.imwrite(filepng, inp['face_bgr'][index_f])

            data_storage = {"ldm": (box, point, lm68) if lm68 else (box, point),
                            "idx": index,
                            "box": start}
            np.savez(filenpz, **data_storage)

        return inp

    def __exit__(self, type, value, tb):
        self.list_track = dict()

# --- util_read.py code ---
class ReadingVideo:
    
    def __init__(self, filename, stride=0):
        self.filename = filename
        self.video_cap = None
        self.stride = stride
        self.length = 0
        self.count = 0
        
    def get_number_frames(self):
        return self.length
    
    def get_shape(self):
        return (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        
    def get_fps(self):
        return self.video_cap.get(cv2.CAP_PROP_FPS)
    
    def __enter__(self):
        self.count = 0
        self.video_cap = cv2.VideoCapture(self.filename)
        self.length = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.stride <= 0:
            ret, img = self.video_cap.read()
            if not ret:
                raise StopIteration
            image_ind = self.count
            self.count = self.count+1
            return {'frame_bgr': img, 'image_ind': image_ind}
        else:
            out = {'frames_bgr': list(), 'frames_inds': list()}
            for _ in range(self.stride):
                ret, img = self.video_cap.read()
                if not ret:
                    break
                out['frames_bgr'].append(img)
                out['frames_inds'].append(self.count)
                self.count = self.count+1
            
            if len(out['frames_bgr']) == 0:
                raise StopIteration
            
            return out
    
    def __call__(self, index):
        return next(self)
    
    def __len__(self):
        return self.length//self.stride + ((self.length % self.stride) > 0)
    
    def __exit__(self, type, value, tb):
        try:
            self.video_cap.release()
        except:
            pass
        self.video_cap = None


def BGR2RGBs(imgs):
    return np.stack([cv2.cvtColor(x.copy(), cv2.COLOR_BGR2RGB) for x in imgs], 0)


class Resampling:
    
    def __init__(self, out_fps=25, list_data=['boxes', ], key_indexs='image_inds'):
        self.list_data = list_data
        self.int_fps = float(out_fps)
        self.out_fps = float(out_fps)
        self.key_indexs = key_indexs
        assert self.key_indexs not in self.list_data
    
    def reset(self, int_fps):
        self.int_fps = float(int_fps)
        return self
    
    def compute_ids(self, ids):
        iout = range(max(int(np.floor((ids-0.5)*self.out_fps/self.int_fps + 1)), 0),
                     int(np.floor((ids+0.5)*self.out_fps/self.int_fps) + 1))
        return list(iout)
    
    def __call__(self, inp):
        out = {key: list() for key in self.list_data}
        out[self.key_indexs] = list()
        
        for index, ids in enumerate(inp[self.key_indexs]):
            ids_outs = self.compute_ids(ids)
            for i in ids_outs:
                out[self.key_indexs].append(i)
                for key in self.list_data:
                    out[key].append(inp[key][index])

        return out


class ReadingResampledVideo:

    def __init__(self, filename, out_fps, stride=1):
        self.filename = filename
        self.video_cap = None
        self.stride = stride
        self.out_fps = float(out_fps)
        self.in_fps = 0
        self.out_length = 0
        self.in_length = 0
        self.in_count = 0
        assert self.stride > 0

    def get_number_frames(self):
        return self.out_length

    def get_shape(self):
        return (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def get_fps(self):
        return self.out_fps

    def __enter__(self):
        self.video_cap = cv2.VideoCapture(self.filename)
        self.in_count = 0
        self.in_length = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.in_fps = float(self.video_cap.get(cv2.CAP_PROP_FPS))
        self.out_length = int(self.in_length * self.out_fps / self.in_fps)
        return self

    def __iter__(self):
        return self

    def compute_ids(self, ids):
        iout = range(max(int(np.floor((ids - 0.5) * self.out_fps / self.in_fps + 1)), 0),
                     int(np.floor((ids + 0.5) * self.out_fps / self.in_fps) + 1))
        return list(iout)

    def __next__(self):
        out = {'frames_bgr': list(), 'frames_inds': list()}

        while len(out['frames_inds']) < self.stride:
            ret, img = self.video_cap.read()
            if not ret:
                break
            ids_outs = self.compute_ids(self.in_count)
            for i in ids_outs:
                out['frames_inds'].append(i)
                out['frames_bgr'].append(img)
            self.in_count = self.in_count + 1

        if len(out['frames_inds']) == 0:
            raise StopIteration

        return out

    def __call__(self, index):
        return next(self)

    def __len__(self):
        return self.out_length // self.stride + ((self.out_length % self.stride) > 0)

    def __exit__(self, type, value, tb):
        try:
            self.video_cap.release()
        except:
            pass
        self.video_cap = None


class MockFileBoxes:
    
    def __init__(self, fileboxes, list_data=['boxes', 'image_inds']):
        if isinstance(fileboxes, list):
            dat = [np.load(_, allow_pickle=True) for _ in fileboxes]
            self.image_inds = np.int64(dat[0]['image_inds'])
            for x, d in zip(fileboxes, dat):
                if not np.array_equal(self.image_inds, np.int64(d['image_inds'])):
                    print('error', x, self.image_inds.shape, d['image_inds'].shape)
                    #os.remove(x)
                    assert False
            self.data = dict()
            for key in list_data:
                found = False
                for d in dat:
                    if key in d:
                        self.data[key] = d[key]
                        found = True
                        break
                if not found:
                    print(f"Warning: key '{key}' not found in any input file. Filling with empty array.")
                    self.data[key] = np.array([])
            del dat
        else:
            dat = np.load(fileboxes, allow_pickle=True)
            self.data = {}
            for key in list_data:
                if key in dat:
                    self.data[key] = dat[key]
                else:
                    print(f"Warning: key '{key}' not found in {fileboxes}. Filling with empty array.")
                    self.data[key] = np.array([])
            self.image_inds = dat['image_inds'] if 'image_inds' in dat else np.array([])
            del dat

    def reset(self):
        return self

    def __len__(self):
        return max(self.image_inds)+1
    
    def __enter__(self):
        return self
    
    def __call__(self, inp):
        if isinstance(inp, dict): 
            if 'frames_inds' in inp:
                ids = inp['frames_inds']
            else:
                ids = [inp['image_ind'], ]
                del inp['image_ind']
        elif isinstance(inp, list):
            ids = inp
            inp = dict()
        else:
            ids = [inp, ]
            inp = dict()
        
        val = [x in ids for x in self.image_inds]
        for key in self.data:
            inp[key] = self.data[key][val]
        
        return inp
    
    def __iter__(self):
        for count in range(len(self)):
            yield self(count)

    def __exit__(self, type, value, tb):
        pass


class FilterTrack:

    def __init__(self, list_data, filt_key, filt_list):
        self.filt_key = filt_key
        self.filt_list = filt_list
        self.list_data = list_data

    def reset(self):
        return self

    def __call__(self, inp):
        ii = [i for i in range(len(inp[self.filt_key])) if inp[self.filt_key][i] in self.filt_list]
        for k in self.list_data:
            inp[k] = [inp[k][i] for i in ii]

        return inp

class FilterValues:

    def __init__(self, condition, list_data=['boxes', 'image_inds', ], key_values='image_inds', list_pass=list()):
        self.list_data = list_data
        self.list_pass = list_pass
        self.condition = condition
        self.key_values = key_values

    def reset(self):
        return self

    def __call__(self, inp):
        out = {key: list() for key in self.list_data}
        for key in self.list_pass:
            out[key] = inp[key]

        for index, ids in enumerate(inp[self.key_values]):
            if self.condition(ids):
                for key in self.list_data:
                    out[key].append(inp[key][index])



        return out
# --- util_face.py code ---

import numpy as np
import cv2


class ReadingVideo:
    
    def __init__(self, filename, stride=0):
        self.filename = filename
        self.video_cap = None
        self.stride = stride
        self.length = 0
        self.count = 0
        
    def get_number_frames(self):
        return self.length
    
    def get_shape(self):
        return (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        
    def get_fps(self):
        return self.video_cap.get(cv2.CAP_PROP_FPS)
    
    def __enter__(self):
        self.count = 0
        self.video_cap = cv2.VideoCapture(self.filename)
        self.length = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.stride <= 0:
            ret, img = self.video_cap.read()
            if not ret:
                raise StopIteration
            image_ind = self.count
            self.count = self.count+1
            return {'frame_bgr': img, 'image_ind': image_ind}
        else:
            out = {'frames_bgr': list(), 'frames_inds': list()}
            for _ in range(self.stride):
                ret, img = self.video_cap.read()
                if not ret:
                    break
                out['frames_bgr'].append(img)
                out['frames_inds'].append(self.count)
                self.count = self.count+1
            
            if len(out['frames_bgr']) == 0:
                raise StopIteration
            
            return out
    
    def __call__(self, index):
        return next(self)
    
    def __len__(self):
        return self.length//self.stride + ((self.length % self.stride) > 0)
    
    def __exit__(self, type, value, tb):
        try:
            self.video_cap.release()
        except:
            pass
        self.video_cap = None


def BGR2RGBs(imgs):
    return np.stack([cv2.cvtColor(x.copy(), cv2.COLOR_BGR2RGB) for x in imgs], 0)


class Resampling:
    
    def __init__(self, out_fps=25, list_data=['boxes', ], key_indexs='image_inds'):
        self.list_data = list_data
        self.int_fps = float(out_fps)
        self.out_fps = float(out_fps)
        self.key_indexs = key_indexs
        assert self.key_indexs not in self.list_data
    
    def reset(self, int_fps):
        self.int_fps = float(int_fps)
        return self
    
    def compute_ids(self, ids):
        iout = range(max(int(np.floor((ids-0.5)*self.out_fps/self.int_fps + 1)), 0),
                     int(np.floor((ids+0.5)*self.out_fps/self.int_fps) + 1))
        return list(iout)
    
    def __call__(self, inp):
        out = {key: list() for key in self.list_data}
        out[self.key_indexs] = list()
        
        for index, ids in enumerate(inp[self.key_indexs]):
            ids_outs = self.compute_ids(ids)
            for i in ids_outs:
                out[self.key_indexs].append(i)
                for key in self.list_data:
                    out[key].append(inp[key][index])

        return out


class ReadingResampledVideo:

    def __init__(self, filename, out_fps, stride=1):
        self.filename = filename
        self.video_cap = None
        self.stride = stride
        self.out_fps = float(out_fps)
        self.in_fps = 0
        self.out_length = 0
        self.in_length = 0
        self.in_count = 0
        assert self.stride > 0

    def get_number_frames(self):
        return self.out_length

    def get_shape(self):
        return (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def get_fps(self):
        return self.out_fps

    def __enter__(self):
        self.video_cap = cv2.VideoCapture(self.filename)
        self.in_count = 0
        self.in_length = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.in_fps = float(self.video_cap.get(cv2.CAP_PROP_FPS))
        self.out_length = int(self.in_length * self.out_fps / self.in_fps)
        return self

    def __iter__(self):
        return self

    def compute_ids(self, ids):
        iout = range(max(int(np.floor((ids - 0.5) * self.out_fps / self.in_fps + 1)), 0),
                     int(np.floor((ids + 0.5) * self.out_fps / self.in_fps) + 1))
        return list(iout)

    def __next__(self):
        out = {'frames_bgr': list(), 'frames_inds': list()}

        while len(out['frames_inds']) < self.stride:
            ret, img = self.video_cap.read()
            if not ret:
                break
            ids_outs = self.compute_ids(self.in_count)
            for i in ids_outs:
                out['frames_inds'].append(i)
                out['frames_bgr'].append(img)
            self.in_count = self.in_count + 1

        if len(out['frames_inds']) == 0:
            raise StopIteration

        return out

    def __call__(self, index):
        return next(self)

    def __len__(self):
        return self.out_length // self.stride + ((self.out_length % self.stride) > 0)

    def __exit__(self, type, value, tb):
        try:
            self.video_cap.release()
        except:
            pass
        self.video_cap = None




class FilterTrack:

    def __init__(self, list_data, filt_key, filt_list):
        self.filt_key = filt_key
        self.filt_list = filt_list
        self.list_data = list_data

    def reset(self):
        return self

    def __call__(self, inp):
        ii = [i for i in range(len(inp[self.filt_key])) if inp[self.filt_key][i] in self.filt_list]
        for k in self.list_data:
            inp[k] = [inp[k][i] for i in ii]

        return inp

class FilterValues:

    def __init__(self, condition, list_data=['boxes', 'image_inds', ], key_values='image_inds', list_pass=list()):
        self.list_data = list_data
        self.list_pass = list_pass
        self.condition = condition
        self.key_values = key_values

    def reset(self):
        return self

    def __call__(self, inp):
        out = {key: list() for key in self.list_data}
        for key in self.list_pass:
            out[key] = inp[key]

        for index, ids in enumerate(inp[self.key_values]):
            if self.condition(ids):
                for key in self.list_data:
                    out[key].append(inp[key][index])



        return out

class DetectFace:
    def __init__(self, device, weights=None, size_threshold=75, target_size=1280,
                 batch_size=16, score_threshold=0.7, iou_threshold=0.5, return_frame=True):
        # from retinaface import get_detector
        from pythonlib.retinaface import get_detector
        self.device = device
        self.batch_size = batch_size
        self.retinaface = get_detector(network='resnet50', device=self.device, trained_model=weights).eval()
        self.target_size = target_size
        self.size_threshold = size_threshold
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.return_frame = return_frame

    def detect_faces(self, images_torch):
        # from retinaface import detect_video_torch
        from pythonlib.retinaface import detect_video_torch
        resize_factor = min(self.target_size / max(images_torch.shape[-1], images_torch.shape[-2]), 1.0)
        with torch.no_grad():
            return detect_video_torch(self.retinaface, images_torch, resize_factor=resize_factor,
                                      score_threshold=self.score_threshold, iou_threshold=self.iou_threshold,
                                      size_threshold=self.size_threshold, batch_size=self.batch_size,
                                      resize_image=resize_factor != 1.0)

    def reset(self):
        return self

    def __call__(self, inp):
        if 'frames_inds' not in inp:
            imgs = [inp['frame_bgr'], ]
            ids = [inp['image_ind'], ]
            del inp['image_ind']
            if not self.return_frame:
                del inp['frame_bgr']
        else:
            imgs = inp['frames_bgr']
            ids = inp['frames_inds']
            if not self.return_frame:
                del inp['frames_bgr']

        imgs = BGR2RGBs(imgs)
        images_torch = (torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to(self.device) - 128) / 128.0

        # face detector
        image_inds, scores, boxes, points = self.detect_faces(images_torch)
        del images_torch
        image_inds = np.asarray([ids[int(x)] for x in image_inds.cpu().numpy()])
        order = np.argsort(image_inds)
        inp['boxes'] = list(boxes.cpu().numpy()[order])
        inp['boxes_score'] = list(scores.cpu().numpy()[order])
        inp['points'] = list(points.cpu().numpy()[order])
        inp['image_inds'] = image_inds[order]

        return inp


def iou(boxes0, boxes1):
    a0 = (boxes0[..., 2] - boxes0[..., 0]) * (boxes0[..., 3] - boxes0[..., 1])
    a1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    w = np.minimum(boxes0[..., 2], boxes1[..., 2]) - np.maximum(boxes0[..., 0], boxes1[..., 0])
    h = np.minimum(boxes0[..., 3], boxes1[..., 3]) - np.maximum(boxes0[..., 1], boxes1[..., 1])
    wh = np.maximum(0.0, w) * np.maximum(0.0, h)
    return wh / (a0 + a1 - wh)


class ComputeTrack:

    def __init__(self, thres=0.5):
        self.lst_boxes = list()
        self.lst_track = list()
        self.thres = thres
        self.detected_tracks = 0
        self.detected_infos = list()

    def reset(self):
        self.lst_boxes = list()
        self.lst_track = list()
        self.detected_tracks = 0
        self.detected_infos = list()
        return self

    def num_tracks(self):
        return self.detected_tracks

    def info_tracks(self):
        return self.detected_infos

    def single_frame(self, frame_ind, new_boxes):

        if len(new_boxes) == 0:
            self.lst_track, self.lst_boxes = list(), list()
            return list()

        new_track = [-1, ] * len(new_boxes)

        if len(self.lst_boxes) == 0:
            assigned_rows = list()
            assigned_cols = list()
            new_cols = list(range(len(new_boxes)))
            ass_scores = [-1, ] * len(new_boxes)
        else:
            scores = iou(np.asarray(new_boxes)[None, :, :], np.asarray(self.lst_boxes)[:, None, :])
            assigned_rows, assigned_cols = linear_sum_assignment(1.0-scores)
            ass_scores = [-1,] * len(new_boxes)
            for a, b in zip(assigned_rows, assigned_cols):
                ass_scores[b] = scores[a, b]
            val = [(scores[a, b] > self.thres) for a, b in zip(assigned_rows, assigned_cols)]
            assigned_rows = [a for a, b in zip(assigned_rows, val) if b]
            assigned_cols = [a for a, b in zip(assigned_cols, val) if b]

            if len(assigned_cols) < len(new_boxes):
                new_cols = [a for a in range(len(new_boxes)) if a not in assigned_cols]
                print(len(new_boxes), scores[:, new_cols])
            else:
                new_cols = list()

        for a, b in zip(assigned_rows, assigned_cols):
            new_track[b] = self.lst_track[a]
            self.detected_infos[new_track[b]][-1] += 1

        for b in new_cols:
            new_track[b] = self.detected_tracks
            self.detected_infos.append([frame_ind, ass_scores[b], 1])
            self.detected_tracks = self.detected_tracks + 1

        self.lst_boxes = new_boxes
        self.lst_track = new_track

        return list(new_track)

    def __call__(self, inp):
        if 'frames_inds' not in inp:
            track = self.single_frame(inp['image_ind'], inp['boxes'])
        else:
            num_boxes = len(inp['boxes'])
            track = [-1, ] * num_boxes
            for frame_ind in inp['frames_inds']:
                pos_boxes1 = [x for x in range(num_boxes) if inp['image_inds'][x] == frame_ind]
                new_boxes1 = [inp['boxes'][x] for x in pos_boxes1]
                track1 = self.single_frame(frame_ind, new_boxes1)
                for i, x in enumerate(pos_boxes1):
                    track[x] = track1[i]

        inp['id_track'] = track
        return inp


class ComputeLandMarkers:

    def __init__(self, device):
        import face_alignment

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)

    def reset(self):
        return self

    def __call__(self, inp):
        # ['face_bgr', 'face_start', 'points', 'boxes']
        if 'face_bgr' in inp:
            imgs = inp['face_bgr']
            face_start = inp['face_start']

        else:
            imgs = inp['frames_bgr']
            image_inds = inp['frames_inds']
            image_inds = [image_inds.index(x) for x in inp['image_inds']]
            imgs = [imgs[x] for x in image_inds]
            del image_inds
            face_start = [(0.0, 0.0), ] * len(imgs)

        boxes = inp['boxes']
        inp['landmarks68'] = list()

        for index in range(len(imgs)):
            img = imgs[index][:, :, ::-1]
            ss = face_start[index]
            box = np.asarray(boxes[index]) - [ss[0], ss[1], ss[0], ss[1]]

            preds = self.fa.get_landmarks(img, detected_faces=[box, ])[0]
            preds = preds + [[ss[0], ss[1]], ]
            inp['landmarks68'].append(preds)

        return inp
# --- grip_unina/extraction.py code ---
def extract_boxes(filevideo, device, opt, verbose=True):

    op2 = DetectFace(device, os.path.join(opt['resources_path'], 'Resnet50_Final.pth'),
                     size_threshold=opt['face_det']['size_threshold'], batch_size=opt['rec_stride'],
                     score_threshold=opt['face_det']['score_threshold'], return_frame=False)
    op3 = ComputeTrack(opt['face_det']['iou_threshold'])

    with ReadingResampledVideo(filevideo, opt['fps'], opt['read_stride']) as video:
        if verbose:
            print(f'Reading video {filevideo} of {video.get_number_frames()} '
                  f'frames with {video.get_fps()} fps.')

        ops = [video, op2.reset(),  op3.reset(), ]
        list_times = [0 for _ in range(len(ops))]
        if verbose:
            print('', flush=True)
            pbar = tqdm(total=len(video))

        dict_out = dict()
        count = 0
        while True:
            try:
                out = count
                for index_op in range(len(ops)):
                    tic = time()
                    out = ops[index_op](out)
                    toc = time()
                    if verbose and (count == 0):
                        print(f"{index_op} step memory: {[(key, len(out[key])) for key in out]}")
                    list_times[index_op] += toc - tic
            except StopIteration:
                break
            count = count + 1
            # if count==3: break
            for key in out.keys():
                if len(out[key]) == 0:
                    continue
                if key in dict_out:
                    dict_out[key].extend(list(out[key]))
                else:
                    dict_out[key] = list(out[key])
            if verbose:
                pbar.update(1)
                pbar.set_description('%d' % op3.num_tracks())

    if verbose:
        print(f"total time: {list_times} sec")
    info_tracks = op3.info_tracks()
    return dict_out, info_tracks


def generate_clips_tracks(filevideo, fileboxes, outputfolder, device, fps,
                          write_one=False, compute_landmarks=False, list_good_track=None, verbose=True):
    from grip_unina.util_read import ReadingResampledVideo
    from grip_unina.util_detect import FaceExtractor
    from grip_unina.util_write import WritingClips
    from grip_unina.util_face import ComputeLandMarkers
    from grip_unina.util_dist import PassIdentity
    from grip_unina.util_read import FilterValues

    op2 = MockFileBoxes(fileboxes, list_data=['boxes', 'image_inds', 'id_track', 'points', ])
    if list_good_track is None:
        op3 = PassIdentity()
    else:
        op3 = FilterValues(condition=lambda x: x in list_good_track,
                       list_data=['boxes', 'image_inds', 'id_track', 'points',],
                       list_pass=['frames_bgr', 'frames_inds'],
                       key_values='id_track')
    op4 = FaceExtractor(face_size=None, square=True, return_frame=False, factor_border=2)
    if compute_landmarks:
        op5 = ComputeLandMarkers(device)
    else:
        op5 = PassIdentity()

    with ReadingResampledVideo(filevideo, fps, 1) as video:
        with WritingClips(outputfolder, write_one=write_one) as write_video:
            if verbose:
                print(f'Reading video {filevideo} of {video.get_number_frames()} frames with {video.get_fps()} fps.')

            ops = [video, op2.reset(), op3.reset(), op4.reset(), op5.reset(), write_video]
            list_times = [0 for _ in range(len(ops))]
            if verbose:
                print('', flush=True)
                pbar = tqdm(total=len(video))

            count = 0
            while True:
                try:
                    out = count
                    for index_op in range(len(ops)):
                        tic = time()
                        out = ops[index_op](out)
                        toc = time()
                        list_times[index_op] += toc - tic
                except StopIteration:
                    break
                count = count + 1

                if verbose:
                    pbar.update(1)

            if verbose:
                print(f"total time: {list_times} sec")


def generate_video_tracks(filevideo, fileboxes, outputvideo, opt, verbose=True):

    op2 = MockFileBoxes(fileboxes, list_data=['boxes', 'image_inds', 'id_track', 'points', ])
    op3 = GenFrameBoxes(tag_boxes='boxes', return_frame=False)

    with ReadingResampledVideo(filevideo, opt['fps'], opt['read_stride']) as video:
        with WritingVideo(outputvideo, fps=opt['fps'], vid_configure=opt['output_ffmpeg_params']) as write_video:
            if verbose:
                print(f'Reading video {filevideo} of {video.get_number_frames()} frames with {video.get_fps()} fps.')

            ops = [video, op2.reset(), op3.reset(), write_video]
            list_times = [0 for _ in range(len(ops))]
            if verbose:
                print('', flush=True)
                pbar = tqdm(total=len(video))

            count = 0
            while True:
                try:
                    out = count
                    for index_op in range(len(ops)):
                        tic = time()
                        out = ops[index_op](out)
                        toc = time()
                        list_times[index_op] += toc - tic
                except StopIteration:
                    break
                count = count + 1

                if verbose:
                    pbar.update(1)

            if verbose:
                print(f"total time: {list_times} sec")


def add_audio_on_video(inputvideo, inputuadio, outputvideo, verbose=True):
    from skvideo import getFFmpegPath
    if verbose:
        print("FFmpeg path: {}".format(getFFmpegPath()))
    cmd = "%s/ffmpeg -hide_banner -loglevel error -y -i '%s' -i '%s' -map 0:v -map 1:a -c:v copy '%s'" % (
        getFFmpegPath(), inputvideo, inputuadio, outputvideo
    )
    os.system(cmd)
    
def extract_boxes(filevideo, device, opt, verbose=True):

    op2 = DetectFace(device, os.path.join(opt['resources_path'], 'Resnet50_Final.pth'),
                     size_threshold=opt['face_det']['size_threshold'], batch_size=opt['rec_stride'],
                     score_threshold=opt['face_det']['score_threshold'], return_frame=False)
    op3 = ComputeTrack(opt['face_det']['iou_threshold'])

    with ReadingResampledVideo(filevideo, opt['fps'], opt['read_stride']) as video:
        if verbose:
            print(f'Reading video {filevideo} of {video.get_number_frames()} '
                  f'frames with {video.get_fps()} fps.')

        ops = [video, op2.reset(),  op3.reset(), ]
        list_times = [0 for _ in range(len(ops))]
        if verbose:
            print('', flush=True)
            pbar = tqdm(total=len(video))

        count = 0
        dict_out = dict()

        while True:
            try:
                out = count
                for index_op in range(len(ops)):
                    tic = time()
                    out = ops[index_op](out)
                    toc = time()
                    if verbose and (count == 0):
                        print(f"{index_op} step memory: {[(key, len(out[key])) for key in out]}")
                    list_times[index_op] += toc - tic
            except StopIteration:
                break
            count = count + 1
            for key in out.keys():
                if len(out[key]) == 0:
                    continue
                if key in dict_out:
                    dict_out[key].extend(list(out[key]))
                else:
                    dict_out[key] = list(out[key])
            if verbose:
                pbar.update(1)
                pbar.set_description('%d' % op3.num_tracks())

    if verbose:
        print(f"total time: {list_times} sec")
    info_tracks = op3.info_tracks()
    return dict_out, info_tracks

# --- Combined script for reference generation ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        description='Complete script to extract features from reference videos.')
    parser.add_argument('--dir_videos', type=str,
                        help='Input directory with the reference video files (with extensions: .mp4, .avi).')
    parser.add_argument('--dir_poi', type=str,
                        help='Output directory where the extracted features will be saved.')
    parser.add_argument('--resources_path', type=str, default="./resources/",
                        help='Directory with networks weights.')
    parser.add_argument('--models', type=str, default='poiforensics',
                        help='Extraction feature of these models.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Index of GPU to use (set to -1 for not using the GPU).')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of videos analyzed in parallel (set to 0 to disable the parallel).')
    parser.add_argument('--stride', type=int, default=32,
                        help='Number of frames analyzed in parallel (reduce it in case of memory errors).')
    parser.add_argument('--tracker_iou_th', type=float, default=0.4,
                        help='Threshold of face tracking.')
    parser.add_argument('--verbose', type=int, default=1)
    # Additional arguments from main_feat_extractor.py
    parser.add_argument('--file_boxes', type=str, default=None)
    parser.add_argument('--file_spec', type=str, default=None)
    parser.add_argument('--file_3dmm', type=str, default=None)
    parser.add_argument('--file_opt', type=str, default=None)
    parser.add_argument('--file_track', type=str, default=None)
    parser.add_argument('--dir_ref', type=str, default=None)
    parser.add_argument('--dir_faces', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--video_input', type=str, default=None)
    argd = parser.parse_args()

    # --- Video discovery logic (from main_gen_references.py) ---
    if argd.dir_videos:
        listfile = glob.glob(os.path.join(argd.dir_videos, '*.mp4')) + \
                   glob.glob(os.path.join(argd.dir_videos, '*.avi'))
    elif argd.video_input:
        listfile = [argd.video_input]
    else:
        listfile = []

    print('Number of found videos:', len(listfile), flush=True)
    listcmd = list()
    for filepath in listfile:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        for model in argd.models.split(','):
            # --- Feature extraction logic (from main_feat_extractor.py) ---
            device = 'cuda:%s' % argd.gpu if is_available() and int(argd.gpu) >= 0 else 'cpu'
            opt = create_opt(resources_path=argd.resources_path,
                             read_stride=3*argd.stride, rec_stride=argd.stride, det_stride=max(argd.stride//2, 1),
                             face_iou_threshold=argd.tracker_iou_th,
                             model=model)
            print('Running on device: {}'.format(device))
            print(f"input : {filepath}")
            print(opt)
            # Output paths
            outputdir = argd.dir_poi
            file_boxes = f'{outputdir}/feats/{filename}/boxes.npz'
            file_spec = f'{outputdir}/feats/{filename}/spec.npy'
            file_3dmm = f'{outputdir}/feats/{filename}/3dmm.npz'
            file_track = f'{outputdir}/track/track_{filename}.mp4'
            file_opt = f'{outputdir}/app_{model}/opt.yaml'
            dir_ref = f'{outputdir}/app_{model}/{filename}'
            dir_faces = f'{outputdir}/faces/{filename}/'
            # --- Save extraction options ---
            if not os.path.isfile(file_opt):
                import yaml
                os.makedirs(os.path.dirname(file_opt), exist_ok=True)
                with open(file_opt, 'w') as fid:
                    documents = yaml.dump(get_extraction_opt(opt), fid)
            # --- Extract boxes ---
            if not os.path.isfile(file_boxes):
                print(f"\noutput: {file_boxes}")
                os.makedirs(os.path.dirname(file_boxes), exist_ok=True)
                dict_out, info_tracks = extract_boxes(filepath, device, opt, verbose=argd.verbose)
                np.savez(file_boxes, **dict_out)
                print(f"\ndone: {file_boxes}", flush=True)
            # --- Generate video tracks ---
            if not os.path.isfile(file_track):
                print(f"\noutput: {file_track}")
                os.makedirs(os.path.dirname(file_track), exist_ok=True)
                tempvideo = file_track + '_tmp.mp4'
                generate_video_tracks(filepath, file_boxes, tempvideo, opt, verbose=argd.verbose)
                add_audio_on_video(tempvideo, filepath, file_track)
                os.remove(tempvideo)
                print(f"\ndone: {file_track}", flush=True)
            # --- Extract audio features ---
            if not os.path.isfile(file_spec):
                print(f"\noutput: {file_spec}")
                os.makedirs(os.path.dirname(file_spec), exist_ok=True)
                from grip_unina.poi_forensics import extract_spec
                audiodata = extract_spec(filepath, opt, verbose=argd.verbose)
                np.save(file_spec, audiodata)
                print(f"\ndone: {file_spec}", flush=True)
            # --- Extract 3DMM features ---
            if not os.path.isfile(file_3dmm):
                print(f"\noutput: {file_3dmm}")
                os.makedirs(os.path.dirname(file_3dmm), exist_ok=True)
                from grip_unina.id_reveal import extract_3dmm
                # dict_out = extract_3dmm(filepath, file_boxes, device, opt, verbose=argd.verbose)
                dict_out = dict()
                np.savez(file_3dmm, **dict_out)
                print(f"\ndone: {file_3dmm}", flush=True)
            # --- Extract reference features ---
            if not os.path.isdir(dir_ref):
                print(f"\noutput: {dir_ref}")
                typ = opt['model']['type']
                if typ == 'poi_forensics':
                    from grip_unina.poi_forensics import extract_feats_poi_forensics
                    dict_out = extract_feats_poi_forensics(filepath,
                                 file_boxes, file_spec,
                                 device=device, opt=opt, verbose=argd.verbose)
                elif typ == 'id_reveal':
                    from grip_unina.id_reveal import extract_feats_idreavel
                    dict_out = extract_feats_idreavel(file_3dmm,
                                 device=device, opt=opt, verbose=argd.verbose)
                elif typ == 'face_recognition':
                    from grip_unina.face_recognition import extract_feats_facerec
                    dict_out = extract_feats_facerec(filepath,
                                 file_boxes,
                                 device=device, opt=opt, verbose=argd.verbose)
                else:
                    assert False
                if 'embs_track' in dict_out:
                    dict_out = {k: np.asarray(dict_out[k]) for k in dict_out}
                    os.makedirs(dir_ref, exist_ok=True)
                    embs_track = dict_out['embs_track']
                    for t in np.unique(embs_track):
                        dict_out_t = {k: dict_out[k][embs_track == t] for k in dict_out if k != 'embs_track'}
                        np.savez(os.path.join(dir_ref, 'embs_track%d.npz' % t), **dict_out_t)
                print(f"\ndone: {dir_ref}", flush=True)
            # --- Generate face clips ---
            if not os.path.isdir(dir_faces):
                if os.path.isdir(dir_ref):
                    list_good_track = [int(_[10:-4]) for _ in os.listdir(dir_ref) if _.startswith('embs_track')]
                else:
                    list_good_track = None
                print(f"\noutput: {dir_faces}", list_good_track)
                os.makedirs(dir_faces, exist_ok=True)
                generate_clips_tracks(filepath, file_boxes, dir_faces, device=device, fps=opt['fps'],
                                      write_one=True, compute_landmarks=False, list_good_track=list_good_track, verbose=argd.verbose)
                print(f"\ndone: {dir_faces}", flush=True)

    print('Reference generation complete.')
# --- End of combined script ---
# Comments indicate where logic from main_gen_references.py and main_feat_extractor.py was merged.
# All dependencies are imported at the top for clarity.
