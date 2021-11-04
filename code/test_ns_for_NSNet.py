import os
import sys
from ns_enhance_onnx import load_onnx_model, denoise_nsnet2

ipath = 'G:\SmartWalker\SSL\dataset\self\\16khz\街市\中\\20210927150015.wav'
opath = 'C:\\Users\wang0\Desktop\\20210927150015.wav'

model, _ = load_onnx_model(model_path='./ns_nsnet2-20ms-baseline.onnx')
denoise_nsnet2(audio_ipath=ipath, audio_opath=opath, model=model, )
