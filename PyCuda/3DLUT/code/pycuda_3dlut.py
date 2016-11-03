import sys
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import pandas as pd
import re


const_matrix_param = np.array([0.2126, 0.7152, 0.0722])


def load_3dlut_cube(filename):
    # LU_3D_SIZE を確認しつつデータ開始行を探索
    # --------------------------------------
    max_header_size = 100
    pattern = re.compile('^\d')
    lut_size = None
    with open(filename, 'r') as fin:
        for idx in range(max_header_size):
            line = fin.readline()
            if lut_size is None:
                if line.find("LUT_3D_SIZE") >= 0:
                    lut_size = line.rstrip().split(" ")[1]
            else:
                if pattern.search(line) is not None:
                    data_start_line = idx + 1

    print("lut_size = {}x{}x{}".format(lut_size, lut_size, lut_size))
    print("start_line = {}".format(data_start_line))


def img_open_and_normalize(filename):
    """
    # 概要
    以下の処理を実施。

    * ファイルのオープン
    * 正規化
    """

    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        print("ERROR!\n{} is not found".format(filename))
        sys.exit(1)

    try:
        img_max_value = np.iinfo(img.dtype).max
    except:
        img_max_value = 1.0

    img = np.float32(img/img_max_value)

    return img


def calc_block_and_grid(nx, ny, dimx, dimy):
    """
    # 概要
    block_size(thread数)とgrid_size(block数)を計算する

    # 参考資料
    CUDA プロフェッショナルプログラミング P.62～P.64
    """

    block = (dimx, dimy, 1)
    grid = ((nx + block[0] - 1) // block[0],
            (ny + block[1] - 1) // block[1])

    return block, grid


def exec_3dlut():

    # 加工元の画像データを準備
    # -----------------------------------
    filename = '../Matrix/figure/src_img.png'
    img = img_open_and_normalize(filename=filename)

    # block数, thread数 の計算
    # ------------------------------------
    nx = img.shape[1]
    ny = img.shape[0]
    block, grid = calc_block_and_grid(nx=nx, ny=ny, dimx=32, dimy=32)

    # GPUに画像データを転送
    # ------------------------------------
    img_gpu = cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(img_gpu, img[:, :, ::-1].tobytes())

    # 3DLUTの次元数を設定
    # ------------------------------------
    

    # カーネルの作成
    # ------------------------------------
    mod = SourceModule("""
        __global__ void matrix_test(float *img, int nx, int ny)
        {
            float r, g, b;
            unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
            unsigned int idx = iy * nx + ix;

            if (ix < nx && iy < ny){
                int r_idx = idx * 3;
                int g_idx = idx * 3 + 1;
                int b_idx = idx * 3 + 2;
                r = img[r_idx];
                g = img[g_idx];
                b = img[b_idx];

                img[r_idx] = r * 0.2126 + g * 0.7152 + b * 0.0722;
                img[g_idx] = r * 0.2126 + g * 0.7152 + b * 0.0722;
                img[b_idx] = r * 0.2126 + g * 0.7152 + b * 0.0722;
            }
        }
        """)

    # カーネルの実行
    # ------------------------------------
    func = mod.get_function("matrix_test")
    func(img_gpu, np.uint32(nx), np.uint32(ny), grid=grid, block=block)

    # 結果の取得
    # ------------------------------------
    img_result = np.empty_like(img)
    cuda.memcpy_dtoh(img_result, img_gpu)
    img_result = img_result[:, :, ::-1]

    # 結果の表示
    # ------------------------------------
    cv2.imshow('preview', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mono_conv_matrix():

    # 加工元の画像データを準備
    # -----------------------------------
    filename = '../Matrix/figure/src_img.png'
    img = img_open_and_normalize(filename=filename)

    # block数, thread数 の計算
    # ------------------------------------
    nx = img.shape[1]
    ny = img.shape[0]
    block, grid = calc_block_and_grid(nx=nx, ny=ny, dimx=32, dimy=32)

    # GPUに画像データを転送
    # ------------------------------------
    img_gpu = cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(img_gpu, img[:, :, ::-1].tobytes())

    # カーネルの作成
    # ------------------------------------
    mod = SourceModule("""
        __global__ void matrix_test(float *img, int nx, int ny)
        {
            float r, g, b;
            unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
            unsigned int idx = iy * nx + ix;

            if (ix < nx && iy < ny){
                int r_idx = idx * 3;
                int g_idx = idx * 3 + 1;
                int b_idx = idx * 3 + 2;
                r = img[r_idx];
                g = img[g_idx];
                b = img[b_idx];

                img[r_idx] = r * 0.2126 + g * 0.7152 + b * 0.0722;
                img[g_idx] = r * 0.2126 + g * 0.7152 + b * 0.0722;
                img[b_idx] = r * 0.2126 + g * 0.7152 + b * 0.0722;
            }
        }
        """)

    # カーネルの実行
    # ------------------------------------
    func = mod.get_function("matrix_test")
    func(img_gpu, np.uint32(nx), np.uint32(ny), grid=grid, block=block)

    # 結果の取得
    # ------------------------------------
    img_result = np.empty_like(img)
    cuda.memcpy_dtoh(img_result, img_gpu)
    img_result = img_result[:, :, ::-1]

    # 結果の表示
    # ------------------------------------
    cv2.imshow('preview', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pass

