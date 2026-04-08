import os
from torch.cuda import is_available
import argparse
import numpy as np

from config import create_opt, get_extraction_opt
# Removed all grip_unina video and bounding box imports 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--resources_path', type=str, default="./resources/")
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--video_input', type=str)
    # All face/3D/track related arguments removed
    parser.add_argument('--file_spec', type=str, default=None)
    parser.add_argument('--file_opt', type=str, default=None)
    parser.add_argument('--model', type=str, default='poiforensics')
    parser.add_argument('--dir_ref', type=str, default=None)
    argd = parser.parse_args()

    device = 'cuda:%s' % argd.gpu if is_available() and int(argd.gpu) >= 0 else 'cpu'
    
    # tracker_iou_th is removed from opt configuration
    opt = create_opt(resources_path=argd.resources_path,
                     read_stride=3*argd.stride, rec_stride=argd.stride, det_stride=max(argd.stride//2, 1),
                     model=argd.model)

    print('Running on device: {}'.format(device))
    print(f"input : {argd.video_input}")
    assert os.path.isfile(argd.video_input)
    print(opt)

    if argd.file_opt is not None:
        if not os.path.isfile(argd.file_opt):
            import yaml
            os.makedirs(os.path.dirname(argd.file_opt), exist_ok=True)
            with open(argd.file_opt, 'w') as fid:
                documents = yaml.dump(get_extraction_opt(opt), fid)

    # Replaced redundant video components. Spec execution preserved.
    if argd.file_spec is not None:
        if not os.path.isfile(argd.file_spec):
            print(f"\noutput: {argd.file_spec}")
            os.makedirs(os.path.dirname(argd.file_spec), exist_ok=True)
            from grip_unina.poi_forensics import extract_spec
            audiodata = extract_spec(argd.video_input, opt, verbose=argd.verbose)
            np.save(argd.file_spec, audiodata)
            print(f"\ndone: {argd.file_spec}", flush=True)

    if argd.dir_ref is not None:
        if not os.path.isdir(argd.dir_ref):
            print(f"\noutput: {argd.dir_ref}")
            typ = opt['model']['type']
            if typ == 'poi_forensics':
                
                # Specifically sourcing a file with dashes using importlib
                import importlib.util
                import sys

                spec_file = "grip_unina/poi_forensics/extraction-copy.py"
                module_name = "audio_only_extractor"
                spec = importlib.util.spec_from_file_location(module_name, spec_file)
                audio_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = audio_module
                spec.loader.exec_module(audio_module)
                
                # Pass into our exclusively audio branch bypassing boxes
                dict_out = audio_module.extract_feats_poi_forensics_audio_only(
                             argd.video_input, argd.file_spec,
                             device=device, opt=opt, verbose=argd.verbose)
            else:
                print("Only poi_forensics model is allowed in the audio-only pipeline.")
                assert False

            if 'embs_track' in dict_out:
                dict_out = {k: np.asarray(dict_out[k]) for k in dict_out}
                os.makedirs(argd.dir_ref, exist_ok=True)
                embs_track = dict_out['embs_track']

                for t in np.unique(embs_track):
                    dict_out_t = {k: dict_out[k][embs_track == t] for k in dict_out if k != 'embs_track'}
                    np.savez(os.path.join(argd.dir_ref, 'embs_track%d.npz' % t), **dict_out_t)
            print(f"\ndone: {argd.dir_ref}", flush=True)
