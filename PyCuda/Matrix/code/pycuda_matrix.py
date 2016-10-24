import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2


const_matrix_param = np.array([0.2126, 0.7152, 0.0722])


def mono_conv_matrix():

    # 加工元の画像データを準備
    # -----------------------------------
    filename = './figure/src_img.png'
    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    try:
        img_max_value = np.iinfo(img.dtype).max
    except:
        img_max_value = 1.0

    img = np.float32(img/img_max_value)

    # block数, thread数 の計算
    # ------------------------------------
    nx = img.shape[1]
    ny = img.shape[0]
    dimx = 32
    dimy = 32
    block = (dimx, dimy, 1)
    grid = ((nx + block[0] - 1) // block[0],
            (ny + block[1] - 1) // block[1])

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
