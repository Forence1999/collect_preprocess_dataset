import os
import joblib
import numpy as np
import soundfile as sf
import subprocess
import glob
from pathlib import Path
import librosa
import random
import tempfile

import pickle
import shutil
from tqdm import tqdm
from threading import Thread
from multiprocessing import Process

from lib.audiolib import audioread, audiowrite, normalize_single_channel_audio, audio_segmenter_4_file, \
    audio_segmenter_4_numpy, audio_energy_ratio_over_threshold, audio_energy_over_threshold, next_greater_power_of_2
from lib.utils import get_files_by_suffix, get_dirs_by_prefix, get_subdirs_by_suffix, get_subdirs_by_suffix, \
    get_subfiles_by_suffix, plot_curve, add_prefix_and_suffix_4_basename
from ns_enhance_onnx import load_onnx_model, denoise_nsnet2
from ssl_feature_extractor import audioFeatureExtractor

ref_audio, _ = audioread('../reference_wav.wav')
REF_AUDIO = normalize_single_channel_audio(ref_audio)
REF_AUDIO_THRESHOLD = (REF_AUDIO ** 2).sum() / len(REF_AUDIO) / 500
EPS = np.finfo(float).eps
REF_POWER = 1e-12
np.random.seed(0)


def decode_dir(d_path):
    '''
    get the basename of a path, and return the '_'-based split of it
    :param d_path:
    :return:
    '''
    return os.path.basename(d_path).split('_')


def decode_file_basename(file_path, seg=False):
    file_name = os.path.basename(os.path.normpath(file_path))
    file_split = file_name.split('_')[1:5]
    if seg:
        seg = [file_name[:-4].split('seg')[-1]]
        return list(map(float, file_split + seg))
    else:
        return list(map(float, file_split))


def decode_src_name(src_name):
    return src_name.split('_')[1:3]


def decode_walker_name(walker_name, doa=True):
    if doa:
        return walker_name.split('_')[1:4], walker_name.split('_')[-1]
    elif not doa:
        return walker_name.split('_')[1:4]


def print_info_of_walker_and_sound_source(dataset_path):
    print('-' * 20, 'info of smart walker', '-' * 20, )
    wk_dirs = get_dirs_by_prefix(dataset_path, 'walker_')
    wk_x, wk_y, wk_z, = set(), set(), set(),
    for i in wk_dirs:
        _, temp_x, temp_y, temp_z = decode_dir(i)
        wk_x.add(temp_x)
        wk_y.add(temp_y)
        wk_z.add(temp_z)
    wk_x, wk_y, wk_z, = list(map(int, wk_x)), list(map(int, wk_y)), list(map(int, wk_z)),
    print('x:', sorted(wk_x), '\n', 'y:', sorted(wk_y), '\n', 'z:', sorted(wk_z), )
    
    print('-' * 20 + 'info of sound source' + '-' * 20, )
    ss_dirs = get_dirs_by_prefix(dataset_path, 'src_')
    s_x, s_y, = set(), set(),
    for i in ss_dirs:
        _, temp_x, temp_y, = decode_dir(i)
        s_x.add(temp_x)
        s_y.add(temp_y)
    s_x, s_y, = list(map(int, s_x)), list(map(int, s_y)),
    print('x:', sorted(s_x), '\n', 'y:', sorted(s_y), )
    
    print('-' * 20 + 'info of direction of arrival (doa)' + '-' * 20, )
    wk_dirs = get_dirs_by_prefix(dataset_path, 'walker_')
    doa = set()
    for i in wk_dirs:
        subdirs = get_subdirs_by_suffix(i, )
        for j in subdirs:
            temp_doa = os.path.basename(j)
            doa.add(temp_doa)
    doa = list(map(int, doa))
    print('doa:', sorted(doa), )


def plot_map(dataset_path):
    '''
    原先为hole数据集而写，为了可视化房间地图，现在已经 be deprecated
    :param dataset_path:
    :return:
    '''
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


def clip_audio(src_dspath, des_dspath, seg_len, stepsize, fs=16000, window='hann', pow_2=False):
    '''
    Clip the audio into segments to segment_len in secs and save them into dir_name
    :param src_dspath: 长片段语音所在的数据集根目录
    :param des_dspath: 处理后数据集的根目录
    :param seg_len: clip 的长度 (单位为 s)
    :param stepsize: 相邻clip间的步长大小(单位 s)
    :param fs: 采样率
    :param window: 为 clip 加窗
    :return: 无返回值
    '''
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    files = get_files_by_suffix(src_dspath, '.wav')
    
    def single_Process(files, ):
        for file in tqdm(files):
            # calculate the save_dpath
            src_doa_dir = os.path.dirname(file)
            rel_path = os.path.relpath(src_doa_dir, start=src_dspath, )
            dst_doa_dir = os.path.join(des_dspath, rel_path, )
            # print('src_doa_dir:', src_doa_dir)
            # print('dst_doa_dir:', dst_doa_dir)
            # segment the audio
            audio_segmenter_4_file(file, dst_doa_dir, segment_len=seg_len, stepsize=stepsize, fs=fs, window=window,
                                   padding=False, pow_2=pow_2, save2segFolders=True)
            
            # # 验证每一个seg folder中均有4通道信号
            # for sub_dpath in get_subdirs_by_suffix(dst_doa_dir):
            #     try:
            #         assert len(get_subfiles_by_suffix(sub_dpath, suffix='.wav')) == 4
            #     except Exception as e:
            #         print('e:', e)
    
    processes = []
    num_process = 32
    for i in range(num_process):
        processes.append(Process(target=single_Process, args=(files[i::num_process],)))
        processes[-1].start()
        # print(f'Process_{i} started', )
    for process in processes:
        process.join()


def preprocessing_audio_with_norm_denoise_drop(src_dspath, fs=16000, threshold=None):
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    files = get_files_by_suffix(src_dspath, '.wav')
    
    def single_Process(files, ):
        denoise_model, _ = load_onnx_model(model_path='./ns_nsnet2-20ms-baseline.onnx')
        for fpath in tqdm(files):
            ini_audio, ini_fs = audioread(fpath)
            assert ini_fs == fs
            
            audio = np.array(ini_audio)
            # norm
            norm_audio, norm_scalar = normalize_single_channel_audio(audio, returnScalar=True)
            # denoise
            de_norm_audio = denoise_nsnet2(audio=norm_audio, fs=fs, model=denoise_model, )
            # drop
            if audio_energy_over_threshold(de_norm_audio, threshold=REF_AUDIO_THRESHOLD) and \
                    audio_energy_ratio_over_threshold(de_norm_audio, fs=fs, threshold=threshold, ):
                doDrop = False
            else:
                doDrop = True
            
            if not doDrop:
                dst_fpath = fpath.replace('ini_hann', 'ini_hann_norm_denoise_drop')
                assert dst_fpath != fpath
                os.makedirs(os.path.dirname(dst_fpath), exist_ok=True)
                de_audio = de_norm_audio / norm_scalar
                audiowrite(destpath=dst_fpath, audio=de_audio, sample_rate=fs, norm=False, clipping_threshold=None, )
    
    processes = []
    num_process = 128
    for i in range(num_process):
        processes.append(Process(target=single_Process, args=(files[i::num_process],)))
        processes[-1].start()
        # print(f'Process_{i} started', )
    for process in processes:
        process.join()


def clean_audio_clips(ds_path):
    files = get_files_by_suffix(ds_path, '.wav')
    
    def single_Process(files, ):
        for fpath in tqdm(files):
            seg_dir = os.path.dirname(fpath)
            seg_files = get_files_by_suffix(seg_dir, '.wav')
            if len(seg_files) < 4:
                try:
                    shutil.rmtree(seg_dir, ignore_errors=True, )
                    print('seg_dir:', seg_dir)
                except:
                    pass
    
    processes = []
    num_process = 64
    for i in range(num_process):
        processes.append(Process(target=single_Process, args=(files[i::num_process],)))
        processes[-1].start()
        # print(f'Process_{i} started', )
    for process in processes:
        process.join()


def preprocessing_audio_with_norm_denoise_drop_norm(src_dspath):
    'dataset -> ds  ;  data -> dt  ;    file -> f  ;   dir -> d  '
    files = get_files_by_suffix(src_dspath, '.wav')
    
    def single_Process(files, ):
        for fpath in tqdm(files):
            ini_audio, ini_fs = audioread(fpath)
            assert ini_fs == fs
            dst_fpath = fpath.replace('ini_hann_norm_denoise_drop', 'ini_hann_norm_denoise_drop_norm')
            assert dst_fpath != fpath
            
            audiowrite(destpath=dst_fpath, audio=ini_audio, sample_rate=ini_fs, norm=True, clipping_threshold=None, )
    
    processes = []
    num_process = 64
    for i in range(num_process):
        processes.append(Process(target=single_Process, args=(files[i::num_process],)))
        processes[-1].start()
        # print(f'Process_{i} started', )
    for process in processes:
        process.join()


def extract_features(src_dspath, dst_dspath, feature_type, num_gcc_bin=128,
                     fft_seg_len=None, fft_stepsize_ratio=None, fs=16000, ):
    feature_ls = ['gcc_phat', 'stft']
    assert feature_type in feature_ls, f'{feature_type} is not supported yet (only support {feature_ls} now).'
    files = get_files_by_suffix(src_dspath, '.wav')
    
    def single_Process(files, ):
        fe = audioFeatureExtractor(num_channel=4, fs=fs, num_gcc_bin=num_gcc_bin, fft_seg_len=fft_seg_len,
                                   fft_stepsize_ratio=fft_stepsize_ratio, datatype='mic', )
        for fpath in tqdm(files):
            # get dst_fpath to save feature
            rel_path = os.path.relpath(fpath, start=src_dspath, )
            dst_fpath = os.path.join(dst_dspath, rel_path, )
            assert dst_fpath != fpath
            dst_fpath = dst_fpath[:-5] + 's.npz'
            if os.path.exists(dst_fpath):  # prevent processing the same audio repeatedly
                continue
            
            audio = []
            mic_path_ls = sorted(get_subfiles_by_suffix(root=os.path.dirname(fpath), suffix='.wav'))
            assert len(mic_path_ls) == 4
            for channel_path in mic_path_ls:
                channel_audio, channel_fs = audioread(channel_path)
                assert channel_fs == fs
                audio.append(channel_audio)
            audio = np.asarray(audio)
            
            # get gcc_feature
            if feature_type == 'gcc_phat':
                feature = fe.get_gcc_phat(audio=audio)
            elif feature_type == 'stft':
                feature = fe.get_stft(audio=audio, )
            else:
                raise ValueError(f'{feature_type} is not supported yet')
            
            # save feature
            os.makedirs(os.path.dirname(dst_fpath), exist_ok=True)
            np.savez(file=dst_fpath, data=feature)
    
    processes = []
    num_process = 128
    for i in range(num_process):
        processes.append(Process(target=single_Process, args=(files[i::num_process],)))
        processes[-1].start()
        # print(f'Process_{i} started', )
    for process in processes:
        process.join()


if __name__ == '__main__':
    print('-' * 20 + 'Preprocessing the dateset' + '-' * 20)
    dataset_root = '../dataset/4F_CYC'
    dataset_ini = os.path.join(dataset_root, 'initial')
    
    print_info_of_walker_and_sound_source(dataset_ini)
    
    segment_para_set = {
        '32ms' : {
            'name'          : '32ms',
            'time_len'      : 32 / 1000,
            'threshold'     : 100,
            'stepsize_ratio': 0.5
        },
        '50ms' : {
            'name'          : '50ms',
            'time_len'      : 50 / 1000,
            'threshold'     : 100,
            'stepsize_ratio': 0.5
        },
        '64ms' : {
            'name'          : '64ms',
            'time_len'      : 64 / 1000,
            'threshold'     : 100,
            'stepsize_ratio': 0.5
        },
        '128ms': {
            'name'          : '128ms',
            'time_len'      : 128 / 1000,
            'threshold'     : 200,  # 100?
            'stepsize_ratio': 0.5
        },
        '256ms': {
            'name'          : '256ms',
            'time_len'      : 256 / 1000,
            'threshold'     : 400,
            'stepsize_ratio': 256 / 1000 / 2
        },
        '1s'   : {
            'name'          : '1s',
            'time_len'      : 1,
            'threshold'     : 800,
            'stepsize_ratio': 0.5,
        },
    }
    fs = 16000
    window = 'hann'
    assert window == 'hann'  # required by the following code
    clip_len = '1s'
    seg_para = segment_para_set[clip_len]
    seg_para['stepsize'] = seg_para['stepsize_ratio'] * seg_para['time_len']
    pow_2 = False
    seg_ds_name = '_'.join([seg_para['name'], str(round(seg_para['stepsize'], 2)), str(seg_para['threshold']), str(fs)])
    seg_len = int(seg_para['time_len'] * fs)
    if pow_2:
        seg_len = next_greater_power_of_2(seg_len)
    step_size = int(seg_len * seg_para['stepsize'])
    fft_len = seg_len
    print('-' * 20 + 'parameters' + '-' * 20, '\n', seg_para)
    print('seg_ds_name:', seg_ds_name)
    print('Actual para:\n', 'seg_len:', seg_len, '\n', 'step_size:', step_size, '\n', 'fft_len:', fft_len, )
    
    initial_dspath = os.path.join(dataset_root, 'initial')
    
    ini_seg_dspath = os.path.join(dataset_root, seg_ds_name, 'ini_' + window)
    # print('Start seg...')
    # clip_audio(initial_dspath, ini_seg_dspath, seg_para['time_len'], seg_para['stepsize'], fs, window=window,
    #            pow_2=pow_2, )
    # print('Finish seg...')
    
    norm_denoise_drop_dspath = add_prefix_and_suffix_4_basename(ini_seg_dspath, suffix='_norm_denoise_drop')
    # print('Start norm_denoise_drop...')
    # preprocessing_audio_with_norm_denoise_drop(src_dspath=ini_seg_dspath, fs=fs, threshold=float(seg_para['threshold']))
    # print('Finish norm_denoise_drop...')
    
    # print('Start cleaning...')
    # clean_audio_clips(ds_path=norm_denoise_drop_dspath)
    # print('Finish cleaning...')
    
    norm_denoise_drop_norm_dspath = add_prefix_and_suffix_4_basename(ini_seg_dspath, suffix='_norm_denoise_drop_norm')
    # print('Start norm_denoise_drop_norm...')
    # preprocessing_audio_with_norm_denoise_drop_norm(src_dspath=norm_denoise_drop_dspath, )
    # print('Finish norm_denoise_drop_norm...')
    
    ######################### extract features #########################
    num_gcc_bin = 128
    gcc_suffix = '_gcc_phat' + '_bin_' + str(num_gcc_bin)
    
    fft_seg_len = 0.064
    fft_stepsize_ratio = 0.5
    fft_suffix = '_stft' + '_seglen_' + str(int(fft_seg_len * 1000)) + 'ms' \
                 + '_stepsize_ratio_' + str(round(fft_stepsize_ratio, 2))
    
    ################ without normalization ################
    print('-' * 20, 'without normalization', '-' * 20, )
    print('Start gcc ...')
    norm_denoise_drop_gcc_phat_dspath = add_prefix_and_suffix_4_basename(norm_denoise_drop_dspath, suffix=gcc_suffix)
    assert norm_denoise_drop_gcc_phat_dspath != norm_denoise_drop_dspath
    extract_features(src_dspath=norm_denoise_drop_dspath, dst_dspath=norm_denoise_drop_gcc_phat_dspath, fs=fs,
                     feature_type='gcc_phat', num_gcc_bin=num_gcc_bin, )
    print('Finish gcc...')
    
    print('Start STFT...')
    norm_denoise_drop_stft_dspath = add_prefix_and_suffix_4_basename(norm_denoise_drop_dspath, suffix=fft_suffix)
    assert norm_denoise_drop_stft_dspath != norm_denoise_drop_dspath
    extract_features(src_dspath=norm_denoise_drop_dspath, dst_dspath=norm_denoise_drop_stft_dspath, fs=fs,
                     feature_type='stft', fft_seg_len=fft_seg_len, fft_stepsize_ratio=fft_stepsize_ratio)
    print('Finish STFT...')
    
    ################ with normalization ################
    print('-' * 20, 'with normalization', '-' * 20, )
    print('Start gcc...')
    norm_denoise_drop_norm_gcc_phat_dspath = \
        add_prefix_and_suffix_4_basename(norm_denoise_drop_norm_dspath, suffix=gcc_suffix)
    assert norm_denoise_drop_norm_gcc_phat_dspath != norm_denoise_drop_norm_dspath
    extract_features(src_dspath=norm_denoise_drop_norm_dspath, dst_dspath=norm_denoise_drop_norm_gcc_phat_dspath, fs=fs,
                     feature_type='gcc_phat', num_gcc_bin=num_gcc_bin, )
    print('Finish gcc...')
    
    print('Start STFT...')
    norm_denoise_drop_norm_stft_dspath = \
        add_prefix_and_suffix_4_basename(norm_denoise_drop_norm_dspath, suffix=fft_suffix)
    assert norm_denoise_drop_norm_stft_dspath != norm_denoise_drop_norm_dspath
    extract_features(src_dspath=norm_denoise_drop_norm_dspath, dst_dspath=norm_denoise_drop_norm_stft_dspath, fs=fs,
                     feature_type='stft', fft_seg_len=fft_seg_len, fft_stepsize_ratio=fft_stepsize_ratio)
    print('Finish STFT...')
