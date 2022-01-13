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
    for f in f_ls:
        os.remove(f)
        print('del the following files: ', f_ls)


def augment_data(ds_dir, base_doa, ):
    '''依次模仿声源旋转90°来将数据增加至4倍'''
    
    base_doa = str(base_doa)
    dir_ls = utils.get_dirs_by_suffix(ds_dir, base_doa)
    
    for old_dir in tqdm(dir_ls):
        p_dir, ori_base = os.path.dirname(old_dir), os.path.basename(old_dir)
        if ori_base != str(base_doa):
            print('jump:', old_dir)
            continue
        
        src_basename_ls = ['record_mic0.wav', 'record_mic1.wav', 'record_mic2.wav', 'record_mic3.wav', ]
        for i in range(1, 4):
            dst_dir = os.path.join(p_dir, str((int(base_doa) + 90 * i) % 360))
            os.makedirs(dst_dir, exist_ok=True)
            
            dst_basename_ls = src_basename_ls[-i:] + src_basename_ls[:-i]
            print(i, ':\n', src_basename_ls, '\n', dst_basename_ls, '\n')
            for j in range(4):
                src_path = os.path.join(old_dir, src_basename_ls[j])
                dst_path = os.path.join(dst_dir, dst_basename_ls[j])
                print(f'copy {src_path} to {dst_path}')
                shutil.copy(src_path, dst_path)


def imitate_hole_dataset(ds_dir):
    '''
    deprecated: will not use this function any more
    :param ds_dir:
    :return:
    '''
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
    '''
    deprecated: will not use this function any more
    :param ds_dir:
    :return:
    '''
    d_ls = utils.get_dirs_by_prefix(ds_dir, 'walker_')
    for d in d_ls:
        # print(d)
        shutil.rmtree(d)


def rename_walker_dir_name(ds_dir):
    '''
    将 walker_ 文件夹的名字从 walker_X_Z_Y 重命名为 walker_X_Y_Z
    :param ds_dir:
    :return:
    '''
    dir_ls = utils.get_dirs_by_prefix(ds_dir, 'walker_')
    for old_dir_path in dir_ls:
        old_base_name = os.path.basename(old_dir_path)
        walker, x, z, y = old_base_name.split('_')
        new_base_name = '_'.join((walker, x, y, z,))
        new_dir_path = os.path.join(os.path.dirname(old_dir_path), new_base_name, )
        print(f'rename {old_dir_path} to {new_dir_path}')
        os.rename(old_dir_path, new_dir_path)


def modify_directions(ds_dir, src_doa, dst_doa):
    '''采集的数据集为右方向为0°，逆时针依次增大。该函数可将角度修正为任意值（比如，修正为正前方为0°，逆时针依次增大）'''
    
    src_doa, dst_doa = str(src_doa), str(dst_doa)
    dir_ls = utils.get_dirs_by_suffix(ds_dir, src_doa)
    
    for old_dir in tqdm(dir_ls):
        p_dir, ori_base = os.path.dirname(old_dir), os.path.basename(old_dir)
        if ori_base != str(src_doa):
            print('jump:', old_dir)
            continue
        new_dir = os.path.join(p_dir, dst_doa)
        print(f'rename {old_dir} to {new_dir}')
        os.rename(old_dir, new_dir)


if __name__ == '__main__':
    ds_dir = '../dataset/4F_CYC/initial'
    
    # !!! run only once
    # del_record_wav(ds_dir)
    # rename_walker_dir_name(ds_dir)
    # modify_directions(ds_dir, src_doa='90', dst_doa='0', )
    # modify_directions(ds_dir, src_doa='-45', dst_doa='315', )
    
    # augment_data(ds_dir=ds_dir, base_doa=0, )
    # augment_data(ds_dir=ds_dir, base_doa=315, )
    # imitate_hole_dataset(ds_dir)  # deprecated: will not use this function any more
    # del_walker_dir(ds_dir)  # deprecated: will not use this function any more
