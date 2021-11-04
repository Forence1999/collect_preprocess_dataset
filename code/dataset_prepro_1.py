import os
import joblib
import numpy as np
import soundfile as sf
import subprocess
import glob
import librosa
import random
import tempfile

import pickle
import shutil
from lib.audiolib import audioread, audiowrite, normalize_single_channel_to_target_level, audio_segmenter_4_file, \
    audio_segmenter_4_numpy, audio_energy_ratio_over_threshold, audio_energy_over_threshold, next_greater_power_of_2
from lib.utils import get_files_by_suffix, get_dirs_by_prefix, plot_curve
from ns_enhance_onnx import load_onnx_model, denoise_nsnet2
from ssl_feature_extractor import GccGenerator
from tqdm import tqdm
from threading import Thread
from multiprocessing import Process
from ssl_feature_extractor import FeatureExtractor

ref_audio, _ = audioread('../reference_wav.wav')
REF_AUDIO = normalize_single_channel_to_target_level(ref_audio)
REF_AUDIO_THRESHOLD = (REF_AUDIO ** 2).sum() / len(REF_AUDIO) / 500
EPS = np.finfo(float).eps
REF_POWER = 1e-12
np.random.seed(0)


def decode_dir(dir_path):
    dir_name = os.path.basename(dir_path)
    dir_split = dir_name.split('_')[1:3]
    
    return list(map(float, dir_split))


def decode_file_basename(file_path, seg=False):
    file_name = os.path.basename(os.path.normpath(file_path))
    file_split = file_name.split('_')[1:5]
    if seg:
        seg = [file_name[:-4].split('seg')[-1]]
        return list(map(float, file_split + seg))
    else:
        return list(map(float, file_split))


def print_info_of_walker_and_sound_source(dataset_path):
    files = get_files_by_suffix(dataset_path, '.wav')
    dirs = get_dirs_by_prefix(dataset_path, 'src')
    
    file_info = []
    x, y, z, d, = set(), set(), set(), set(),
    for i in files:
        temp = decode_file_basename(i)
        file_info.append(temp)
        x.add(temp[0])
        y.add(temp[1])
        z.add(temp[2])
        d.add(temp[3])
    print('-' * 20 + 'info of smart walker' + '-' * 20, '\n', 'x:', sorted(x), '\n', 'y:', sorted(y), '\n', 'z:',
          sorted(z),
          '\n', 'd:', sorted(d), )
    
    dir_info = []
    s_x, s_y, = set(), set(),
    for i in dirs:
        temp = decode_dir(i)
        dir_info.append(temp)
        s_x.add(temp[0])
        s_y.add(temp[1])
    print('-' * 20 + 'info of sound source' + '-' * 20, '\n', 'x:', sorted(s_x), '\n', 'y:', sorted(s_y), )


def plot_map(dataset_path):
    dirs = get_dirs_by_prefix(dataset_path, 'src')
    
    for dir in dirs:
        print('\n', '-' * 20 + 'room_map' + '-' * 20, )
        
        arrows = ['->', '-°', '|^', '°-', '<-', '.|', '!!', '|.']
        files = get_files_by_suffix(dir, '.wav')
        [s_x, s_y] = list(map(float, os.path.basename(dir).split('_')[1:3]))
        print(s_x, s_y)
        
        room_map = np.ndarray((15, 19), dtype=object, )
        for i in files:
            temp = decode_file_basename(i)
            w_x = int(float(temp[0]) * 2) + 7
            w_z = int(float(temp[2]) * 2) + 9
            w_doa = int(temp[3]) // 45
            room_map[w_x, w_z] = arrows[w_doa]
        s_x = int(s_x * 2) + 7
        s_y = int(s_y * 2) + 9
        room_map[s_x, s_y] = 'oo'
        room_map = np.flip(room_map, axis=0)
        room_map = np.flip(room_map, axis=1)
        
        for i in range(len(room_map)):
            print(list(room_map[i]))


def clean_audio_clips(ds_path):
    'delete the audio clips which cannot be paired (four microphones)'
    src_dirs = os.listdir(ds_path)
    
    def single_thread(src_dirs, ):
        for src_name in tqdm(src_dirs):
            for walker_name in os.listdir(os.path.join(ds_path, src_name)):
                dir_path = os.path.join(ds_path, src_name, walker_name)
                files = get_files_by_suffix(dir_path, '.wav')
                seg_num = [int(os.path.basename(i)[:-4].split('seg')[-1]) for i in files]
                for i in range(min(seg_num), max(seg_num) + 1):
                    seg_files = get_files_by_suffix(dir_path, 'seg' + str(i) + '.wav')
                    # print(len(seg_files))
                    if len(seg_files) < 4:
                        # print(seg_files)
                        for seg_file in seg_files:
                            os.remove(seg_file)
    
    processes = []
    for i in range(10):
        processes.append(Process(target=single_thread, args=(src_dirs[i::10],)))
        processes[-1].start()
    for process in processes:
        process.join()


def segment_audio(dataset_root, dir_name, seg_len, stepsize=1., fs=16000, window='hann'):
    '''Segment the audio clips to segment_len in secs and save them into dir_name'''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = os.path.join(dataset_root, 'initial')
    des_dspath = os.path.join(dataset_root, dir_name, 'ini_' + window)
    
    files = get_files_by_suffix(ini_dspath, '.wav')
    
    def single_thread(files, ):
        for file in tqdm(files):
            # calculate the save_dpath
            file_dir = os.path.dirname(file)
            file_name, _ = os.path.splitext(os.path.basename(file))
            file_name = '_'.join(file_name.split('_')[:-1])
            rel_path = os.path.relpath(file_dir, start=ini_dspath, )
            save_dpath = os.path.join(des_dspath, rel_path, file_name)
            
            # segment the audio
            audio_segmenter_4_file(file, save_dpath, segment_len=seg_len, stepsize=stepsize, fs=fs, window=window,
                                   padding=False, pow_2=True)
    
    processes = []
    for i in range(20):
        processes.append(Process(target=single_thread, args=(files[i::20],)))
        processes[-1].start()
        print('\nProcess %d has been created' % i)
    for process in processes:
        process.join()


def denoise_audio_clips(dataset_root, dir_name, ini_ds_name, des_ds_name):
    '''Denoise the audio clips and save them into dir_name'''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = os.path.join(dataset_root, dir_name, ini_ds_name)
    save_dspath = os.path.join(dataset_root, dir_name, des_ds_name)
    
    files = get_files_by_suffix(ini_dspath, '.wav')
    
    def single_thread(files, ):
        model, _ = load_onnx_model(model_path='./ns_nsnet2-20ms-baseline.onnx')
        for file in tqdm(files):
            # calculate the save_dpath
            file_name, _ = os.path.splitext(os.path.basename(file))
            rel_path = os.path.relpath(file, start=ini_dspath, )
            save_dpath = os.path.join(save_dspath, rel_path, )
            
            # denoise the audio
            denoise_nsnet2(audio_ipath=file, audio_opath=save_dpath, model=model, )
    
    processes = []
    for i in range(50):
        processes.append(Process(target=single_thread, args=(files[i::50],)))
        processes[-1].start()
        print('\nProcess %d has been created' % i)
    for process in processes:
        process.join()


def normalize_audio_clips(dataset_root, dir_name, ini_ds_name, des_ds_name):
    '''Denoise the audio clips and save them into dir_name'''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = os.path.join(dataset_root, dir_name, ini_ds_name, )  # 'denoise_nsnet2'
    save_dspath = os.path.join(dataset_root, dir_name, des_ds_name)  # 'normalized_denoise_nsnet2'
    
    files = get_files_by_suffix(ini_dspath, '.wav')
    
    def single_thread(files, ):
        for file in tqdm(files):
            # calculate the save_dpath
            file_name, _ = os.path.splitext(os.path.basename(file))
            rel_path = os.path.relpath(file, start=ini_dspath, )
            save_dpath = os.path.join(save_dspath, rel_path, )
            
            # normalize the audio
            # denoise_nsnet2(audio_ipath=file, audio_opath=save_dpath, model=model, )
            audio, fs = audioread(file)
            audiowrite(save_dpath, audio, sample_rate=fs, norm=True, target_level=-25, clipping_threshold=0.99)
    
    processes = []
    for i in range(50):
        processes.append(Process(target=single_thread, args=(files[i::50],)))
        processes[-1].start()
        print('\nProcess %d has been created' % i)
    for process in processes:
        process.join()


def decode_audio_path(path):
    '''decode the info of the path of one audio
    ss: sound source
    wk: walker
    '''
    path = os.path.normpath(path)
    [wk_x, wk_z, wk_y, doa, seg] = decode_file_basename(path, seg=True)
    
    file_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    file_split = file_name.split('_')[1:3]
    [src_x, src_y] = list(map(float, file_split))
    
    return [src_x, src_y, wk_x, wk_y, wk_z, doa, seg]


def decode_src_name(src_name):
    return src_name.split('_')[1:3]


def decode_walker_name(walker_name, doa=True):
    if doa:
        return walker_name.split('_')[1:4], walker_name.split('_')[-1]
    elif not doa:
        return walker_name.split('_')[1:4]


def pack_data_into_dict(dataset_root, dir_name, ini_ds_name):
    '''pack the data into the dictionary form'''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = os.path.join(dataset_root, dir_name, ini_ds_name)
    save_dspath = os.path.join(dataset_root, dir_name, ini_ds_name + '_dict.pkl')
    os.makedirs(os.path.dirname(save_dspath), exist_ok=True)
    
    dataset = {}
    
    for src_name in tqdm(os.listdir(ini_dspath)):
        # print('-' * 20, 'Processing ', src_name, '-' * 20, )
        src_key = '_'.join(decode_src_name(src_name))
        dataset[src_key] = {}
        for walker_name in os.listdir(os.path.join(ini_dspath, src_name)):
            walker_key, doa = decode_walker_name(walker_name, doa=True)
            walker_key = '_'.join(walker_key)
            if walker_key in list(dataset[src_key].keys()):
                pass
            else:
                dataset[src_key][walker_key] = {}
            dataset[src_key][walker_key][doa] = {}
            
            dir_path = os.path.join(ini_dspath, src_name, walker_name)
            files = get_files_by_suffix(dir_path, '.wav')
            seg_set = set([int(os.path.basename(i)[:-4].split('seg')[-1]) for i in files])
            # print(dir_path, '\n', seg_set)
            for seg in seg_set:
                seg_files = sorted(get_files_by_suffix(dir_path, 'seg' + str(seg) + '.wav'))
                # print(len(seg_files))
                seg_audio_list = []
                for file in seg_files:
                    audio, _ = audioread(file)
                    seg_audio_list.append(audio)
                dataset[src_key][walker_key][doa][str(seg)] = np.array([seg_audio_list])
    
    time_len, stepsize, threshold, fs = dir_name.split('_')
    save_ds = {
        'dataset'             : dataset,
        'fs'                  : int(fs),
        'time_len'            : time_len,
        'stepsize'            : str(stepsize) + 's',
        'threshold'           : threshold,
        'data_proprecess_type': '_'.join((dir_name, ini_ds_name)),
        'description'         : 'The dataset is organized as follows:\n\
                                dataset[src_key][wk_key][doa][str(seg)] = np.array([seg_audio_list])\n\
                                i.e. src: sound source | wk: walker | doa: direction of arrival | seg: segment number of the audio clip in the original long audio | \n\
                                src_key: src_x, src_y | wk_key: wk_x, wk_y, wk_z\n\
                                x, y, z: horizontal coordinate! , vertical coordinate! , hight of the walker (always be 1 in this dataset)\n\
                                fs: sample rate\n\
                                time_len: time length of one clip\n\
                                stepsize: stepsize when clipping the original audio\n\
                                threshold: threshold when dropping the clips\n\
                                data_proprecess_type: different preprocessing stage of this dataset'
    }
    with open(save_dspath, 'wb') as fo:
        pickle.dump(save_ds, fo, protocol=4)


def pack_data_into_array(dataset_root, dir_name, ini_ds_name):
    '''pack the data into the numpy-array form'''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = os.path.join(dataset_root, dir_name, ini_ds_name)
    save_dspath = os.path.join(dataset_root, dir_name, ini_ds_name + '_np.pkl')
    os.makedirs(os.path.dirname(save_dspath), exist_ok=True)
    
    ds_audio_list, ds_label_list = [], []
    for src_name in tqdm(os.listdir(ini_dspath)):
        # print('-' * 20, 'Processing ', src_name, '-' * 20, )
        src_audio_list, src_label_list = [], []
        for walker_name in os.listdir(os.path.join(ini_dspath, src_name)):
            dir_path = os.path.join(ini_dspath, src_name, walker_name)
            files = get_files_by_suffix(dir_path, '.wav')
            seg_set = set([int(os.path.basename(i)[:-4].split('seg')[-1]) for i in files])
            # print(dir_path, '\n', seg_set)
            for i in seg_set:
                seg_files = sorted(get_files_by_suffix(dir_path, 'seg' + str(i) + '.wav'))
                # print(len(seg_files))
                seg_audio_list, seg_label_list = [], []
                for file in seg_files:
                    # print(os.path.basename(file))
                    audio, _ = audioread(file)
                    seg_audio_list.append(audio)
                    seg_label_list.append(decode_audio_path(file))  # decode the info of the audio
                
                src_audio_list.append([seg_audio_list])
                src_label_list.append(seg_label_list[0])
        ds_audio_list.append(np.array(src_audio_list))
        ds_label_list.append(np.array(src_label_list))
    ds_audio_array = np.array(ds_audio_list, dtype=object)
    ds_label_array = np.array(ds_label_list, dtype=object)
    
    time_len, stepsize, threshold, fs = dir_name.split('_')
    
    dataset = {
        'x'                   : ds_audio_array,
        'y'                   : ds_label_array,
        'fs'                  : int(fs),
        'time_len'            : time_len,
        'stepsize'            : str(stepsize) + 's',
        'threshold'           : threshold,
        'data_proprecess_type': '_'.join((dir_name, ini_ds_name)),
        'description'         : 'The first dimension of this dataset is organized based on the locations of different sound sources. \n\
                                And the following dimensions are sample_number * 1 * microphones (i.e. 4) * sample_points. \n\
                                The label is [src_x, src_y, wk_x, wk_y, wk_z, doa, seg] \n\
                                i.e. src: sound source | wk: walker | doa: direction of arrival | seg: segment number of the audio clip in the original long audio | \n\
                                x , y , z:  horizontal coordinate! , vertical coordinate! , hight of the walker (always be 1 in this dataset)\n\
                                fs: sample rate\n\
                                time_len: time length of one clip\n\
                                stepsize: overlap_ratio when clipping the original audio\n\
                                threshold: threshold when dropping the clips\n\
                                data_proprecess_type: different preprocessing stage of this dataset'
    }
    with open(save_dspath, 'wb') as fo:
        pickle.dump(dataset, fo, protocol=4)


@staticmethod
def next_lower_power_of_2(x):
    return 2 ** ((int(x)).bit_length() - 1)


def extract_gcc_phat_from_np(dataset_path, fs, fft_len, feature_len=128):
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = dataset_path
    save_dspath = os.path.join(ini_dspath[:-4] + '_gcc_phat_' + str(feature_len) + '.pkl')
    os.makedirs(os.path.dirname(save_dspath), exist_ok=True)
    
    with open(ini_dspath, 'rb') as fo:
        dataset = pickle.load(fo)
    
    x = dataset['x']
    feature_extractor = FeatureExtractor(fs=fs, fft_len=fft_len, num_gcc_bin=feature_len, num_mel_bin=feature_len,
                                         datatype='mic', )
    gcc_ds_ls = []
    for i in range(len(x)):
        gcc_src_ls = []
        for j in range(len(x[i])):
            audio_ls = x[i][j]
            gcc_seg_ls = feature_extractor.get_gcc_phat(audio=audio_ls)
            gcc_src_ls.append([gcc_seg_ls])
        gcc_ds_ls.append(np.array(gcc_src_ls))
    gcc_ds_array = np.array(gcc_ds_ls, dtype=object)
    
    gcc_dataset = {
        'x'                   : gcc_ds_array,
        'y'                   : dataset['y'],
        'fs'                  : dataset['fs'],
        'time_len'            : dataset['time_len'],
        'stepsize'            : dataset['stepsize'],
        'feature_len'         : feature_len,
        'threshold'           : dataset['threshold'],
        'data_proprecess_type': dataset['data_proprecess_type'],
        'description'         : 'The first dimension of this dataset is organized based on the location of different sound sources. \n\
        And the following dimensions are sample_number * 1* combinations (the different combinations of two microphones, i.e. C*2_4 = 6) * gcc_features( i.e. 128). \n\
        The label is [src_x, src_y, wk_x, wk_y, wk_z, doa, seg] \n\
         i.e. src: sound source | wk: walker | doa: direction of arrival | seg: segment number of the audio clip in the original long audio | \n\
         x , y , z:  horizontal coordinate! , vertical coordinate! , hight of the walker (always be 1 in this dataset)\n\
         fs: sample rate\n\
         time_len: time length of one clip\n\
         stepsize: overlap_ratio when clipping the original audio\n\
         feature_len: length of feature\n\
         threshold: threshold when dropping the clips\n\
         data_proprecess_type: different preprocessing stage of this dataset'
    }
    with open(save_dspath, 'wb') as fo:
        pickle.dump(gcc_dataset, fo, protocol=4)


def extract_gcc_phat_from_dict(dataset_path, fs, fft_len, feature_len=128):
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = dataset_path
    save_dspath = os.path.join(ini_dspath[:-4] + '_gcc_phat_' + str(feature_len) + '.pkl')
    os.makedirs(os.path.dirname(save_dspath), exist_ok=True)
    
    with open(ini_dspath, 'rb') as fo:
        dataset = pickle.load(fo)
    
    gcc_dataset = dataset['dataset']
    feature_extractor = FeatureExtractor(fs=fs, fft_len=fft_len, num_gcc_bin=feature_len, num_mel_bin=feature_len,
                                         datatype='mic', )
    for src_key in list(gcc_dataset.keys()):
        for wk_key in list(gcc_dataset[src_key].keys()):
            for doa_key in list(gcc_dataset[src_key][wk_key].keys()):
                for seg_key in list(gcc_dataset[src_key][wk_key][doa_key].keys()):
                        audio_ls = gcc_dataset[src_key][wk_key][doa_key][seg_key]
                        gcc_seg_ls = feature_extractor.get_gcc_phat(audio=audio_ls)
                        gcc_dataset[src_key][wk_key][doa_key][seg_key] = gcc_seg_ls
    
    gcc_dataset = {
        'dataset'             : gcc_dataset,
        'fs'                  : dataset['fs'],
        'time_len'            : dataset['time_len'],
        'stepsize'            : dataset['stepsize'],
        'feature_len'         : feature_len,
        'threshold'           : dataset['threshold'],
        'data_proprecess_type': dataset['data_proprecess_type'],
        'description'         : 'The dataset is organized as follows:\n\
                                dataset[src_key][wk_key][doa][str(seg)] = np.array([seg_gcc_phat])\n\
                                i.e. src: sound source | wk: walker | doa: direction of arrival | seg: segment number of the audio clip in the original long audio | \n\
                                src_key: src_x, src_y | wk_key: wk_x, wk_y, wk_z\n\
                                x, y, z: horizontal coordinate! , vertical coordinate! , hight of the walker (always be 1 in this dataset)\n\
                                fs: sample rate\n\
                                time_len: time length of one clip\n\
                                stepsize: stepsize when clipping the original audio\n\
                                threshold: threshold when dropping the clips\n\
                                data_proprecess_type: different preprocessing stage of this dataset'
    }
    with open(save_dspath, 'wb') as fo:
        pickle.dump(gcc_dataset, fo, protocol=4)


def extract_log_mel_from_np(dataset_path, fs, fft_len, feature_len=128):
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = dataset_path
    save_dspath = os.path.join(ini_dspath[:-4] + '_log_mel_' + str(feature_len) + '.pkl')
    os.makedirs(os.path.dirname(save_dspath), exist_ok=True)
    
    with open(ini_dspath, 'rb') as fo:
        dataset = pickle.load(fo)
    
    x = dataset['x']
    feature_extractor = FeatureExtractor(fs=fs, fft_len=fft_len, num_gcc_bin=feature_len, num_mel_bin=feature_len,
                                         datatype='mic', )
    mel_ds_ls = []
    for i in range(len(x)):
        mel_src_ls = []
        for j in range(len(x[i])):
            audio_ls = x[i][j]
            mel_seg_ls = feature_extractor.get_log_mel(audio=audio_ls)
            mel_src_ls.append([mel_seg_ls])
        mel_ds_ls.append(np.array(mel_src_ls))
    mel_ds_array = np.array(mel_ds_ls, dtype=object)
    
    mel_dataset = {
        'x'                   : mel_ds_array,
        'y'                   : dataset['y'],
        'fs'                  : dataset['fs'],
        'time_len'            : dataset['time_len'],
        'stepsize'            : dataset['stepsize'],
        'feature_len'         : feature_len,
        'threshold'           : dataset['threshold'],
        'data_proprecess_type': dataset['data_proprecess_type'],
        'description'         : 'The first dimension of this dataset is organized based on the location of different sound sources. \n\
        And the following dimensions are sample_number * 1* combinations (the different combinations of two microphones, i.e. C*2_4 = 6) * log_mel_features( i.e. 128). \n\
        The label is [src_x, src_y, wk_x, wk_y, wk_z, doa, seg] \n\
         i.e. src: sound source | wk: walker | doa: direction of arrival | seg: segment number of the audio clip in the original long audio | \n\
         x , y , z:  horizontal coordinate! , vertical coordinate! , hight of the walker (always be 1 in this dataset)\n\
         fs: sample rate\n\
         time_len: time length of one clip\n\
         stepsize: overlap_ratio when clipping the original audio\n\
         feature_len: length of feature\n\
         threshold: threshold when dropping the clips\n\
         data_proprecess_type: different preprocessing stage of this dataset'
    }
    with open(save_dspath, 'wb') as fo:
        pickle.dump(mel_dataset, fo, protocol=4)


def extract_log_mel_from_dict(dataset_path, fs, fft_len, feature_len=128):
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = dataset_path
    save_dspath = os.path.join(ini_dspath[:-4] + '_log_mel_' + str(feature_len) + '.pkl')
    os.makedirs(os.path.dirname(save_dspath), exist_ok=True)
    
    with open(ini_dspath, 'rb') as fo:
        dataset = pickle.load(fo)
    
    mel_dataset = dataset['dataset']
    feature_extractor = FeatureExtractor(fs=fs, fft_len=fft_len, num_gcc_bin=feature_len, num_mel_bin=feature_len,
                                         datatype='mic', )
    for src_key in list(mel_dataset.keys()):
        for wk_key in list(mel_dataset[src_key].keys()):
            for doa_key in list(mel_dataset[src_key][wk_key].keys()):
                for seg_key in list(mel_dataset[src_key][wk_key][doa_key].keys()):
                    audio_ls = mel_dataset[src_key][wk_key][doa_key][seg_key]
                    mel_seg_ls = feature_extractor.get_log_mel(audio=audio_ls)
                    mel_dataset[src_key][wk_key][doa_key][seg_key] = mel_seg_ls

    mel_dataset = {
        'dataset'             : mel_dataset,
        'fs'                  : dataset['fs'],
        'time_len'            : dataset['time_len'],
        'stepsize'            : dataset['stepsize'],
        'feature_len'         : feature_len,
        'threshold'           : dataset['threshold'],
        'data_proprecess_type': dataset['data_proprecess_type'],
        'description'         : 'The dataset is organized as follows:\n\
                                dataset[src_key][wk_key][doa][str(seg)] = np.array([seg_log_mel])\n\
                                i.e. src: sound source | wk: walker | doa: direction of arrival | seg: segment number of the audio clip in the original long audio | \n\
                                src_key: src_x, src_y | wk_key: wk_x, wk_y, wk_z\n\
                                x, y, z: horizontal coordinate! , vertical coordinate! , hight of the walker (always be 1 in this dataset)\n\
                                fs: sample rate\n\
                                time_len: time length of one clip\n\
                                stepsize: stepsize when clipping the original audio\n\
                                threshold: threshold when dropping the clips\n\
                                data_proprecess_type: different preprocessing stage of this dataset'
    }

    with open(save_dspath, 'wb') as fo:
        pickle.dump(mel_dataset, fo, protocol=4)


def drop_audio_clips(dataset_root, dir_name, ini_ds_name, des_ds_name, fs, threshold):
    '''Denoise the audio clips and save them into dir_name'''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    
    ini_dspath = os.path.join(dataset_root, dir_name, ini_ds_name, )  # 'denoise_nsnet2'
    save_dspath = os.path.join(dataset_root, dir_name, des_ds_name)  # 'normalized_denoise_nsnet2'
    
    files = get_files_by_suffix(ini_dspath, '.wav')
    
    def single_thread(files, ):
        for file in tqdm(files):
            # calculate the save_dpath
            file_name, _ = os.path.splitext(os.path.basename(file))
            rel_path = os.path.relpath(file, start=ini_dspath, )
            save_dpath = os.path.join(save_dspath, rel_path, )
            
            # normalize the audio
            # denoise_nsnet2(audio_ipath=file, audio_opath=save_dpath, model=model, )
            audio, fs = audioread(file)
            if audio_energy_over_threshold(audio, threshold=REF_AUDIO_THRESHOLD) and \
                    audio_energy_ratio_over_threshold(audio, fs=fs, threshold=threshold, ):
                os.makedirs(os.path.dirname(save_dpath), exist_ok=True)
                shutil.copy(file, save_dpath)
            else:
                continue
    
    processes = []
    for i in range(20):
        processes.append(Process(target=single_thread, args=(files[i::20],)))
        processes[-1].start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    print('-' * 20 + 'Preprocessing the dateset' + '-' * 20)
    dataset_root = '../dataset/4F_CYC'
    dataset_ini = os.path.join(dataset_root, 'initial')
    
    print_info_of_walker_and_sound_source(dataset_ini)
    
    # plot_map(dataset_ini)
    
    segment_para_set = {
        '32ms' : {
            'name'     : '32ms',
            'time_len' : 32 / 1000,
            'threshold': 100,
            'stepsize' : 0.5
        },
        '50ms' : {
            'name'     : '50ms',
            'time_len' : 50 / 1000,
            'threshold': 100,
            'stepsize' : 0.5
        },
        '64ms' : {
            'name'     : '64ms',
            'time_len' : 64 / 1000,
            'threshold': 100,
            'stepsize' : 0.5
        },
        '128ms': {
            'name'     : '128ms',
            'time_len' : 128 / 1000,
            'threshold': 200,  # 100?
            'stepsize' : 0.5
        },
        '256ms': {
            'name'     : '256ms',
            'time_len' : 256 / 1000,
            'threshold': 400,
            'stepsize' : 256 / 1000 / 2
        },
        '1s'   : {
            'name'     : '1s',
            'time_len' : 1,
            'threshold': 800,
            'stepsize' : 0.5
        },
    }
    fs = 16000
    window = 'hann'
    seg_para = segment_para_set['256ms']
    dir_name = seg_para['name'] + '_' + str(round(seg_para['stepsize'], 2)) + '_' + str(
        seg_para['threshold']) + '_' + str(fs)
    print('-' * 20 + 'parameters' + '-' * 20, '\n', seg_para)
    print('Actual para:\n',
          'seg_len: ', next_greater_power_of_2(seg_para['time_len'] * fs),
          'step_size: ', next_greater_power_of_2(seg_para['stepsize'] * fs), )
    fft_len = next_greater_power_of_2(seg_para['time_len'] * fs)
    
    # print('Start seg...')
    # segment_audio(dataset_root, dir_name, seg_para['time_len'], seg_para['stepsize'], fs, window=window)
    # print('Finish seg...')
    
    # ''' norm denoise drop clean norm pack '''
    # print('Start norm...')
    # normalize_audio_clips(dataset_root, dir_name=dir_name, ini_ds_name='ini_' + window,
    #                       des_ds_name='norm_ini_' + window)
    # print('Finish norm...')
    
    # print('Start denoising...')
    # denoise_audio_clips(dataset_root, dir_name=dir_name, ini_ds_name='norm_ini_' + window,
    #                     des_ds_name='denoised_norm_ini_' + window)
    # print('Finish denoising...')
    
    # print('Start dropping...')
    # drop_audio_clips(dataset_root, dir_name=dir_name, ini_ds_name='denoised_norm_ini_' + window,
    #                  des_ds_name='drop_denoised_norm_ini_' + window, fs=fs, threshold=seg_para['threshold'])
    # print('Finish dropping...')
    #
    # print('Start cleaning...')
    # clean_audio_clips(ds_path=os.path.join(dataset_root, dir_name, 'drop_denoised_norm_ini_' + window))
    # print('Finish cleaning...')
    #
    # print('Start norm...')
    # normalize_audio_clips(dataset_root, dir_name=dir_name, ini_ds_name='drop_denoised_norm_ini_' + window,
    #                       des_ds_name='norm_drop_denoised_norm_ini_' + window)
    # print('Finish norm...')
    
    # print('Start packing...')
    # pack_data_into_array(dataset_root, dir_name=dir_name, ini_ds_name='norm_drop_denoised_norm_ini_' + window)
    # print('Finish packing...')
    #
    # print('Start packing...')
    # pack_data_into_dict(dataset_root, dir_name=dir_name, ini_ds_name='norm_drop_denoised_norm_ini_' + window)
    # print('Finish packing...')
    #
    ''' denoise drop clean norm pack '''
    #
    # print('Start denoising...')
    # denoise_audio_clips(dataset_root, dir_name=dir_name, ini_ds_name='ini_' + window,
    #                     des_ds_name='denoised_ini_' + window)
    # print('Finish denoising...')
    # #
    # print('Start dropping...')
    # drop_audio_clips(dataset_root, dir_name=dir_name, ini_ds_name='denoised_ini_' + window,
    #                  des_ds_name='drop_denoised_ini_' + window, fs=fs, threshold=seg_para['threshold'])
    # print('Finish dropping...')
    #
    # print('Start cleaning...')
    # clean_audio_clips(ds_path=os.path.join(dataset_root, dir_name, 'drop_denoised_ini_' + window))
    # print('Finish cleaning...')
    #
    # print('Start norm...')
    # normalize_audio_clips(dataset_root, dir_name=dir_name, ini_ds_name='drop_denoised_ini_' + window,
    #                       des_ds_name='norm_drop_denoised_ini_' + window)
    # print('Finish norm...')
    # #
    # print('Start packing...')
    # pack_data_into_array(dataset_root, dir_name=dir_name, ini_ds_name='norm_drop_denoised_ini_' + window)
    # print('Finish packing...')
    #
    # print('Start packing...')
    # pack_data_into_dict(dataset_root, dir_name=dir_name, ini_ds_name='norm_drop_denoised_ini_' + window)
    # print('Finish packing...')
    
    ''' denoise drop clean pack '''
    
    # print('Start denoising...')
    # denoise_audio_clips(dataset_root, dir_name=dir_name, ini_ds_name='ini_' + window,
    #                     des_ds_name='denoised_ini_' + window)
    # print('Finish denoising...')
    #
    # print('Start dropping...')
    # drop_audio_clips(dataset_root, dir_name=dir_name, ini_ds_name='denoised_ini_' + window,
    #                  des_ds_name='drop_denoised_ini_' + window, fs=fs, threshold=seg_para['threshold'])
    # print('Finish dropping...')
    #
    # print('Start cleaning...')
    # clean_audio_clips(ds_path=os.path.join(dataset_root, dir_name, 'drop_denoised_ini_' + window))
    # print('Finish cleaning...')
    
    # print('Start packing...')
    # pack_data_into_array(dataset_root, dir_name=dir_name, ini_ds_name='drop_denoised_ini_' + window)
    # print('Finish packing...')
    #
    # print('Start packing...')
    # pack_data_into_dict(dataset_root, dir_name=dir_name, ini_ds_name='drop_denoised_ini_' + window)
    # print('Finish packing...')
    #
    ''' drop clean pack '''
    
    # print('Start dropping...')
    # drop_audio_clips(dataset_root, dir_name=dir_name, ini_ds_name='ini_' + window,
    #                  des_ds_name='drop_ini_' + window, fs=fs, threshold=seg_para['threshold'])
    # print('Finish dropping...')
    #
    # print('Start cleaning...')
    # clean_audio_clips(ds_path=os.path.join(dataset_root, dir_name, 'drop_ini_' + window))
    # print('Finish cleaning...')
    
    # print('Start packing...')
    # pack_data_into_array(dataset_root, dir_name=dir_name, ini_ds_name='drop_ini_' + window)
    # print('Finish packing...')
    #
    # print('Start packing...')
    # pack_data_into_dict(dataset_root, dir_name=dir_name, ini_ds_name='drop_ini_' + window)
    # print('Finish packing...')
    ''' pack '''
    # print('Start packing...')
    # pack_data_into_array(dataset_root, dir_name=dir_name, ini_ds_name='ini_' + window)
    # print('Finish packing...')
    #
    # print('Start packing...')
    # pack_data_into_dict(dataset_root, dir_name=dir_name, ini_ds_name='ini_' + window)
    # print('Finish packing...')
    
    ''' extract features '''
    # ds_name_base = 'ini_hann'
    # ds_name_base = 'drop_denoised_ini_hann'
    # ds_name_base = 'norm_drop_denoised_ini_hann'
    ds_name_base = 'norm_drop_denoised_norm_ini_hann'
    ds_name_np = ds_name_base + '_np.pkl'
    ds_name_dict = ds_name_base + '_dict.pkl'

    # print('Start gcc...')
    # dataset_path = os.path.join(dataset_root, dir_name, ds_name_np)
    # print(dataset_path)
    # extract_gcc_phat_from_np(dataset_path=dataset_path, fs=fs, fft_len=fft_len, feature_len=128)
    # dataset_path = os.path.join(dataset_root, dir_name, ds_name_dict)
    # print(dataset_path)
    # extract_gcc_phat_from_dict(dataset_path=dataset_path, fs=fs, fft_len=fft_len, feature_len=128)
    # print('Finish gcc...')

    print('Start log_mel...')
    # # dataset_path = os.path.join(dataset_root, dir_name, ds_name_np)
    # print(dataset_path)
    # # extract_log_mel_from_np(dataset_path=dataset_path, fs=fs, fft_len=fft_len, feature_len=128)
    dataset_path = os.path.join(dataset_root, dir_name, ds_name_dict)
    print(dataset_path)
    extract_log_mel_from_dict(dataset_path=dataset_path, fs=fs, fft_len=fft_len, feature_len=128)
    print('Finish log_mel...')
