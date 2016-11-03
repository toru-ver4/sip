import sys
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import re


const_matrix_param = np.array([0.2126, 0.7152, 0.0722])


def load_3dlut_cube(filename):
    """
    # 概要
    CUBE形式の3DLUTデータ読み込み Numpy形式で返す。
    CUBE形式については以下URLの「3DLUT の 読み込みと書き込み」を参照。
        https://goo.gl/yJmZRH

    # 注意事項
    不正ファイルを読み込んだ場合の処理とか全く無いからね。
    変なファイルを読み込ませないでね。
    """
    # LUT_3D_SIZE を確認しつつデータ開始行を探索
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
                    skip_line = idx
                    break

    print("lut_size = {}x{}x{}".format(lut_size, lut_size, lut_size))
    print("data start line = {}".format(skip_line + 1))

    # np.loadtxt で numpy形式にして読み込み
    # --------------------------------------
    lut_data = np.loadtxt(fname=filename,
                          dtype=np.float32, delimiter=' ', skiprows=skip_line)

    return lut_data


def save_3dlut_cube(lut_data, filename):
    """
    # 概要
    CUBE形式の3DLUTデータをファイルに書き出す。
    CUBE形式については以下URLの「3DLUT の 読み込みと書き込み」を参照。
        https://goo.gl/yJmZRH

    # 注意事項
    適当に作った関数なので、行数が＋１行されててもエラー出ないからね。
    エラー処理を期待しないでね。
    """
    grid_num = np.uint8(np.round(np.power(lut_data.shape[0], 1/3)))

    with open(filename, 'w') as fout:
        fout.write("# THIS IS MADE BY TORU YOSHIHARA\n\n")
        fout.write("DOMAIN_MIN {} {} {}\n".format(0, 0, 0))
        fout.write("DOMAIN_MAX {} {} {}\n".format(1, 1, 1))
        fout.write("LUT_3D_SIZE {}\n".format(grid_num))
        fout.write("\n")
        for data in lut_data:
            fout.write("{:.11f} {:.11f} {:.11f}\n".format(
                data[0], data[1], data[2]))


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

