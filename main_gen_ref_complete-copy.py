# %%writefile /tmp/script.py

import os
import argparse
import glob
import tqdm
import yaml
import numpy as np
import sys
sys.path.append('/kaggle/input/datasets/tausifr/pythonlib-folder/')
sys.path.append('/kaggle/input/datasets/tausifr/grip-folder4/')
sys.path.append('/kaggle/input/datasets/tausifr/conf-folder/')


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
        # file_model = './config/%s.yaml' % opt['model']
        file_model = '/kaggle/input/datasets/tausifr/conf-folder/config/%s.yaml' % opt['model']

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

def process_video(task):
    from torch.cuda import is_available
    import importlib.util


    filepath = task['filepath']
    file_spec = task['file_spec']
    model = task['model']
    dir_ref = task['dir_ref']
    file_opt = task['file_opt']
    gpu = task['gpu']
    verbose = task['verbose']
    resources_path = task['resources_path']
    stride = task['stride']

    device = 'cuda:%s' % gpu if is_available() and int(gpu) >= 0 else 'cpu'

    opt = create_opt(resources_path=resources_path,
                     read_stride=3*stride, rec_stride=stride, det_stride=max(stride//2, 1),
                     model=model)

    if verbose:
        print('Running on device: {}'.format(device))
        print(f"input : {filepath}")
        print(opt)

    if file_opt is not None:
        if not os.path.isfile(file_opt):
            os.makedirs(os.path.dirname(file_opt), exist_ok=True)
            with open(file_opt, 'w') as fid:
                yaml.dump(get_extraction_opt(opt), fid)

    if file_spec is not None:
        if not os.path.isfile(file_spec):
            if verbose: print(f"\noutput: {file_spec}")
            os.makedirs(os.path.dirname(file_spec), exist_ok=True)
            from grip_unina.poi_forensics import extract_spec
            audiodata = extract_spec(filepath, opt, verbose=verbose)
            np.save(file_spec, audiodata)
            if verbose: print(f"\ndone: {file_spec}", flush=True)

    if dir_ref is not None:
        if not os.path.isdir(dir_ref):
            if verbose: print(f"\noutput: {dir_ref}")
            typ = opt['model']['type']
            if typ == 'poi_forensics':
                
                # Specifically sourcing a file with dashes using importlib
                # spec_file = "grip_unina/poi_forensics/extraction-copy.py"
                spec_file = "/kaggle/input/datasets/tausifr/grip-folder4/grip_unina/poi_forensics/extraction-copy.py"
                module_name = "audio_only_extractor"
                spec = importlib.util.spec_from_file_location(module_name, spec_file)
                audio_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = audio_module
                spec.loader.exec_module(audio_module)
                
                # Pass into our exclusively audio branch bypassing boxes
                dict_out = audio_module.extract_feats_poi_forensics_audio_only(
                             filepath, file_spec,
                             device=device, opt=opt, verbose=verbose)
            else:
                print("Only poi_forensics model is allowed in the audio-only pipeline.")
                return

            if 'embs_track' in dict_out:
                dict_out = {k: np.asarray(dict_out[k]) for k in dict_out}
                os.makedirs(dir_ref, exist_ok=True)
                embs_track = dict_out['embs_track']

                for t in np.unique(embs_track):
                    dict_out_t = {k: dict_out[k][embs_track == t] for k in dict_out if k != 'embs_track'}
                    np.savez(os.path.join(dir_ref, 'embs_track%d.npz' % t), **dict_out_t)
            if verbose: print(f"\ndone: {dir_ref}", flush=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        description='script to extract features from the reference videos.')
    # KAGGLE NOTE: Change to input directory (e.g., /kaggle/input/your-dataset/videos/)
    parser.add_argument('--dir_videos', type=str,
                        help='input directory with the reference audio/video files (with extensions: .mp4, .avi, .wav).')
    # KAGGLE NOTE: Must be a writable directory (e.g., /kaggle/working/poi_features/)
    parser.add_argument('--dir_poi', type=str,
                        help='output directory where the extracted features will be saved.')
    # KAGGLE NOTE: Update to your dataset path containing weights (e.g., /kaggle/input/model-weights/)
    parser.add_argument('--resources_path', type=str, default="./resources/",
                        help='directory with networks weights.')
    #default='idreveal,poiforensics'
    parser.add_argument('--models', type=str, default='poiforensics',
                        help='extraction feature of these models.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='index of GPU to use (set to -1 for not using the GPU).')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of videos analyzed in parallel (set to 0 to disable the parallel).')
    parser.add_argument('--stride', type=int, default=32,
                        help='number of frames analyzed in parallel (reduce it in case of memory errors).')

    parser.add_argument('--verbose', type=int, default=1)
    argd = parser.parse_args()

    outputdir = argd.dir_poi
    listfile = glob.glob(os.path.join(argd.dir_videos, '*.mp4')) + \
               glob.glob(os.path.join(argd.dir_videos, '*.avi')) + \
               glob.glob(os.path.join(argd.dir_videos, '*.wav'))

    print('Number of found audio/video files:', len(listfile), flush=True)
    listtasks = list()
    for filepath in listfile:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        for model in argd.models.split(','):
            listtasks.append({
                'filepath': filepath,
                'file_spec': f'{outputdir}/feats/{filename}/spec.npy',
                'model': model,
                'dir_ref': f'{outputdir}/app_{model}/{filename}',
                'file_opt': f'{outputdir}/app_{model}/opt.yaml',
                'gpu': argd.gpu,
                'verbose': argd.verbose,
                'resources_path': argd.resources_path,
                'stride': argd.stride
            })

    if argd.workers < 1:
        for task in tqdm.tqdm(listtasks, total=len(listtasks)):
            process_video(task)
    else:
        from multiprocessing import get_context
        ctx = get_context("fork")
        with ctx.Pool(argd.workers) as pool:
            list(tqdm.tqdm(pool.imap_unordered(process_video, listtasks), total=len(listtasks)))
