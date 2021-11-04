import os
import sys
import wave
import numpy as np
import soundfile as sf
from pyaudio import PyAudio, paInt16
from lib.audiolib import audiowrite, audioread
from lib.utils import standard_normalizaion, add_prefix_and_suffix_4_basename
import lib.utils as utils
import shutil
from tqdm import tqdm


def del_record_wav(ds_dir):
    '''
    delete all the files ended with 'record.wav'
    '''
    f_ls = utils.get_files_by_suffix(ds_dir, 'record.wav')
    print('del the following files: ', f_ls)
    for f in f_ls:
        os.remove(f)


def augment_data(base, ds_dir):
    d_ls = utils.get_dirs_by_suffix(ds_dir, str(base))
    for raw_d in tqdm(d_ls):
        if os.path.basename(raw_d) != str(base):
            print('jump ', raw_d)
            continue
        raw_basename_ls = ['record_mic0.wav', 'record_mic1.wav', 'record_mic2.wav', 'record_mic3.wav', ]
        
        parent_d = os.path.dirname(raw_d)
        for i in range(1, 4):
            des_d = os.path.join(parent_d, str((int(base) + 90 * i) % 360))
            if not os.path.exists(des_d):
                os.makedirs(des_d)
            
            des_basename_ls = raw_basename_ls[-i:] + raw_basename_ls[:-i]
            for j in range(len(raw_basename_ls)):
                raw_path = os.path.join(raw_d, raw_basename_ls[j])
                des_path = os.path.join(des_d, des_basename_ls[j])
                shutil.copy(raw_path, des_path)


def imitate_hole_dataset(ds_dir):
    f_ls = utils.get_files_by_suffix(ds_dir, '.wav')
    
    for f in f_ls:
        p_dir = os.path.dirname(f)
        basename = os.path.basename(f)
        
        pp_dir = os.path.dirname(p_dir)
        direction_info = os.path.basename(p_dir)
        
        src_dir = os.path.dirname(pp_dir)
        walker_info = os.path.basename(pp_dir)
        
        des_path = os.path.join(src_dir, walker_info + '_' + direction_info + basename[-9:])
        # print(f, '\n', des_path)
        shutil.move(f, des_path)


def del_walker_dir(ds_dir):
    d_ls = utils.get_dirs_by_prefix(ds_dir, 'walker_')
    for d in d_ls:
        # print(d)
        shutil.rmtree(d)


if __name__ == '__main__':
    ds_dir = '../dataset/4F_CYC/initial'
    del_record_wav(ds_dir)
    augment_data(base=45, ds_dir=ds_dir)
    augment_data(base=90, ds_dir=ds_dir)
    imitate_hole_dataset(ds_dir)
    del_walker_dir(ds_dir)
