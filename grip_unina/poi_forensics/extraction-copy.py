import os
import numpy as np
from time import time
from tqdm import tqdm


def extract_spec(filevideo, opt, verbose=True):
    from grip_unina.util_audio import compute_spec
    return compute_spec(filevideo,
                     target_sampling_rate=opt['audio']['sampling_rate'],
                     audio_norm_target_dBFS=opt['audio']['norm_target_dBFS'],
                     n_fft=opt['audio']['num_fft'],
                     window_step=opt['audio']['window_step'],
                     window_length=opt['audio']['window_length'])


def get_info(x):
    if isinstance(x, np.ndarray):
        return 'array', x.shape, x.dtype
    elif len(x) > 0:
        if isinstance(x[0], list) or isinstance(x[0], tuple):
            return len(x), 'list', len(x[0])
        elif isinstance(x[0], np.ndarray):
            return len(x), 'array', x[0].shape, x[0].dtype
        else:
            return len(x), 'none'
    else:
        return 0


def extract_feats_poi_forensics_audio_only(filevideo, filespec, device, opt, verbose=True):
    from grip_unina.util_audio import MockFileSpec, IterImageInds
    from grip_unina.poi_forensics.models import load_model
    from grip_unina.util_dist_audiovideo import ComputeTemporalMulti
    from grip_unina.util_model3d import AlignAudio, ApplyModel3d
    from torch import from_numpy as numpy2torch
    
    spec_data = np.load(filespec)
    
    # Generate mock iterator mimicking frame loops for the deepfake detector purely using audio indices.
    factor = opt['fps'] * opt['audio']['window_step']
    datastep = 1000 // factor
    num_frames = spec_data.shape[0] // datastep
    
    op3 = MockFileSpec(spec_data, fps=opt['fps'], audio_window_step=opt['audio']['window_step'], output_key='spec')
    
    align_audio = AlignAudio()
    op5 = ComputeTemporalMulti(opt['model']['clip_length'], opt['model']['clip_stride'],
                               list_elem=align_audio.input_keys(),
                               function=align_audio,
                               outkeys=align_audio.output_keys())
                               
    network_audio, _ = load_model(opt['resources_path'], opt['model'], device)
    
    # Run only audio resnet
    op6 = ApplyModel3d(device, network_audio, batch_size=opt['det_stride'], transform=numpy2torch,
                       input_key='embs_spec', output_key='embs_feat_audio')
    
    with IterImageInds(total=num_frames, stride=opt['read_stride']) as video_iterator:
        if verbose:
            print(f'Simulating media {filevideo} with {num_frames} frames representing audio chunks.')

        ops = [video_iterator, op3.reset(), op5.reset(), op6.reset()]
        list_times = [0 for _ in range(len(ops))]
        if verbose:
            print('', flush=True)
            pbar = tqdm(total=len(video_iterator))

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
                        stat_memory = {key: get_info(out[key]) for key in out}
                        print(f"{index_op} step memory: {stat_memory}")
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

    if verbose:
        print(f"total time: {list_times} sec")

    return dict_out

