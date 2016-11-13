import sys
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import re
import numba


def exec_3dlut_on_gpu(img, lut):

    # grid_num を算出しつつ lut の shape を確認
    # ------------------------------------------
    grid_num = np.uint8(np.round(np.power(lut.shape[0], 1/3)))
    total_index = grid_num**3
    if lut.shape[0] != total_index:
        print("error! 3DLUT size is invalid.")
        return None

    img = img * (grid_num - 1)

    # block数, thread数 の計算
    # ------------------------------------
    nx = img.shape[1]
    ny = img.shape[0]
    block, grid = calc_block_and_grid(nx=nx, ny=ny, dimx=16, dimy=8)

    # GPUに画像データと3DLUTを転送
    # ------------------------------------
    img_gpu = cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(img_gpu, img[:, :, ::-1].tobytes())
    lut_gpu = cuda.mem_alloc(lut.nbytes)
    cuda.memcpy_htod(lut_gpu, lut.tobytes())

    # カーネルの作成
    # ------------------------------------
    mod = SourceModule("""
        __global__ void exec_3dlut(float *img, float *lut, int grid_num, int nx, int ny)
        {
            float r, g, b;
            unsigned int r_idx, g_idx, b_idx;
            unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
            unsigned int idx = iy * nx + ix;
            unsigned int total_index = (int)(pow((double)grid_num, 3.0));

            if (ix < nx && iy < ny){

                // get RGB value
                int r_px_idx = idx * 3;
                int g_px_idx = idx * 3 + 1;
                int b_px_idx = idx * 3 + 2;
                r = img[r_px_idx];
                g = img[g_px_idx];
                b = img[b_px_idx];

                // get 3dlut index
                r_idx = (unsigned int)(floor(r));
                g_idx = (unsigned int)(floor(g));
                b_idx = (unsigned int)(floor(b));

                // calc volume
                float r_0, r_1, g_0, g_1, b_0, b_1;
                float v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7;
                r_0 = r - r_idx;
                r_1 = 1 - r_0;
                g_0 = g - g_idx;
                g_1 = 1 - g_0;
                b_0 = b - b_idx;
                b_1 = 1 - b_0;

                v_0 = (r_0 * g_0 * b_0);
                v_1 = (r_0 * g_0 * b_1);
                v_2 = (r_0 * g_1 * b_0);
                v_3 = (r_0 * g_1 * b_1);
                v_4 = (r_1 * g_0 * b_0);
                v_5 = (r_1 * g_0 * b_1);
                v_6 = (r_1 * g_1 * b_0);
                v_7 = (r_1 * g_1 * b_1);

                // get lut value
                unsigned int r_coef, g_coef, b_coef;
                int l0_idx, l1_idx, l2_idx, l3_idx, l4_idx, l5_idx, l6_idx, l7_idx;
                float *l_0, *l_1, *l_2, *l_3, *l_4, *l_5, *l_6, *l_7;

                r_coef = (int)(pow((double)grid_num, 0.0));
                g_coef = (int)(pow((double)grid_num, 1.0));
                b_coef = (int)(pow((double)grid_num, 2.0));

                l0_idx = ((r_idx + 0)*(r_coef) + (g_idx + 0)*(g_coef) + (b_idx + 0)*(b_coef)) * 3;
                l1_idx = ((r_idx + 0)*(r_coef) + (g_idx + 0)*(g_coef) + (b_idx + 1)*(b_coef)) * 3;
                l2_idx = ((r_idx + 0)*(r_coef) + (g_idx + 1)*(g_coef) + (b_idx + 0)*(b_coef)) * 3;
                l3_idx = ((r_idx + 0)*(r_coef) + (g_idx + 1)*(g_coef) + (b_idx + 1)*(b_coef)) * 3;
                l4_idx = ((r_idx + 1)*(r_coef) + (g_idx + 0)*(g_coef) + (b_idx + 0)*(b_coef)) * 3;
                l5_idx = ((r_idx + 1)*(r_coef) + (g_idx + 0)*(g_coef) + (b_idx + 1)*(b_coef)) * 3;
                l6_idx = ((r_idx + 1)*(r_coef) + (g_idx + 1)*(g_coef) + (b_idx + 0)*(b_coef)) * 3;
                l7_idx = ((r_idx + 1)*(r_coef) + (g_idx + 1)*(g_coef) + (b_idx + 1)*(b_coef)) * 3;

                l_0 = &lut[l0_idx % (total_index * 3)];
                l_1 = &lut[l1_idx % (total_index * 3)];
                l_2 = &lut[l2_idx % (total_index * 3)];
                l_3 = &lut[l3_idx % (total_index * 3)];
                l_4 = &lut[l4_idx % (total_index * 3)];
                l_5 = &lut[l5_idx % (total_index * 3)];
                l_6 = &lut[l6_idx % (total_index * 3)];
                l_7 = &lut[l7_idx % (total_index * 3)];

                // exec 3dlut linear interpolation
                img[r_px_idx] = v_0*l_7[0] + v_1*l_6[0] + v_2*l_5[0] + v_3*l_4[0]
                    + v_4*l_3[0] + v_5*l_2[0] + v_6*l_1[0] + v_7*l_0[0];
                img[g_px_idx] = v_0*l_7[1] + v_1*l_6[1] + v_2*l_5[1] + v_3*l_4[1]
                    + v_4*l_3[1] + v_5*l_2[1] + v_6*l_1[1] + v_7*l_0[1];
                img[b_px_idx] = v_0*l_7[2] + v_1*l_6[2] + v_2*l_5[2] + v_3*l_4[2]
                    + v_4*l_3[2] + v_5*l_2[2] + v_6*l_1[2] + v_7*l_0[2];
            }
        }
        """)

    # カーネルの実行
    # ------------------------------------
    func = mod.get_function("exec_3dlut")
    func(img_gpu, lut_gpu, np.uint32(grid_num), np.uint32(nx),
         np.uint32(ny), grid=grid, block=block)

    # 結果の取得
    # ------------------------------------
    img_result = np.empty_like(img)
    cuda.memcpy_dtoh(img_result, img_gpu)
    img_result = img_result[:, :, ::-1]

    return img_result


@numba.jit
def exec_3dlut_on_x86(img, lut):
    """
    # 概要
    3DLUTを画像に適用する。

    # 詳細
    lut には Adobe CUBE 形式の順序で並んだ 3DLUTデータを
    Numpy の np.float32 で入れておくこと。

    # 備考
    本関数は将来的にCUDAに移植することを考えて
    forループを使って実装している。よってクソ遅い。
    耐えられないようであれば numba.jit のデコレータを使うこと。
    """

    # grid_num を算出しつつ lut の shape を確認
    # ------------------------------------------
    grid_num = np.uint8(np.round(np.power(lut.shape[0], 1/3)))
    total_index = grid_num**3
    if lut.shape[0] != total_index:
        print("error! 3DLUT size is invalid.")
        return None

    img = img * (grid_num - 1)
    out_img = np.empty_like(img)

    for w_idx in range(img.shape[1]):
        for h_idx in range(img.shape[0]):
            r = img[h_idx][w_idx][0]
            g = img[h_idx][w_idx][1]
            b = img[h_idx][w_idx][2]

            # r, g, b の各 Index を求める
            # ---------------------------
            r_idx = np.uint32(np.floor(r))
            g_idx = np.uint32(np.floor(g))
            b_idx = np.uint32(np.floor(b))

            # 体積算出のために r_0 ～ b_1 を求める
            # ------------------------------------
            r_0 = r - r_idx
            r_1 = 1 - r_0
            g_0 = g - g_idx
            g_1 = 1 - g_0
            b_0 = b - b_idx
            b_1 = 1 - b_0

            # 体積計算
            # -------------------------------------
            v_0 = (r_0 * g_0 * b_0)
            v_1 = (r_0 * g_0 * b_1)
            v_2 = (r_0 * g_1 * b_0)
            v_3 = (r_0 * g_1 * b_1)
            v_4 = (r_1 * g_0 * b_0)
            v_5 = (r_1 * g_0 * b_1)
            v_6 = (r_1 * g_1 * b_0)
            v_7 = (r_1 * g_1 * b_1)

            # l_0 ～ l_7 を事前に求めておく
            # --------------------------------------
            r_coef = grid_num ** 0
            g_coef = grid_num ** 1
            b_coef = grid_num ** 2

            l0_idx = (r_idx + 0)*(r_coef) + (g_idx + 0)*(g_coef) + (b_idx + 0)*(b_coef)
            l1_idx = (r_idx + 0)*(r_coef) + (g_idx + 0)*(g_coef) + (b_idx + 1)*(b_coef)
            l2_idx = (r_idx + 0)*(r_coef) + (g_idx + 1)*(g_coef) + (b_idx + 0)*(b_coef)
            l3_idx = (r_idx + 0)*(r_coef) + (g_idx + 1)*(g_coef) + (b_idx + 1)*(b_coef)
            l4_idx = (r_idx + 1)*(r_coef) + (g_idx + 0)*(g_coef) + (b_idx + 0)*(b_coef)
            l5_idx = (r_idx + 1)*(r_coef) + (g_idx + 0)*(g_coef) + (b_idx + 1)*(b_coef)
            l6_idx = (r_idx + 1)*(r_coef) + (g_idx + 1)*(g_coef) + (b_idx + 0)*(b_coef)
            l7_idx = (r_idx + 1)*(r_coef) + (g_idx + 1)*(g_coef) + (b_idx + 1)*(b_coef)

            """
            以下で `% total_index` をしているのは端点対策。
            r, g, b のいずれかが 1.0 だと端点が足りなくなる。
            なので適当なLUTを当てる。
            最終的にそのLUT値は掛け算で 0.0 になるし。
            """
            l_0 = lut[l0_idx % total_index]
            l_1 = lut[l1_idx % total_index]
            l_2 = lut[l2_idx % total_index]
            l_3 = lut[l3_idx % total_index]
            l_4 = lut[l4_idx % total_index]
            l_5 = lut[l5_idx % total_index]
            l_6 = lut[l6_idx % total_index]
            l_7 = lut[l7_idx % total_index]

            # 線形補間処理実行
            # -----------------------------------------
            r_out = v_0*l_7[0] + v_1*l_6[0] + v_2*l_5[0] + v_3*l_4[0]\
                + v_4*l_3[0] + v_5*l_2[0] + v_6*l_1[0] + v_7*l_0[0]
            g_out = v_0*l_7[1] + v_1*l_6[1] + v_2*l_5[1] + v_3*l_4[1]\
                + v_4*l_3[1] + v_5*l_2[1] + v_6*l_1[1] + v_7*l_0[1]
            b_out = v_0*l_7[2] + v_1*l_6[2] + v_2*l_5[2] + v_3*l_4[2]\
                + v_4*l_3[2] + v_5*l_2[2] + v_6*l_1[2] + v_7*l_0[2]

            # 結果を out_img に詰める
            # -----------------------------------------
            out_img[h_idx][w_idx][0] = r_out
            out_img[h_idx][w_idx][1] = g_out
            out_img[h_idx][w_idx][2] = b_out

    return out_img


@numba.jit
def exec_3dlut_on_x86_fast(img, lut):
    """
    # 概要
    3DLUTを画像に適用する。
    初代は処理速度が遅かったので計算量が減るように改善した

    # 詳細
    lut には Adobe CUBE 形式の順序で並んだ 3DLUTデータを
    Numpy の np.float32 で入れておくこと。

    # 備考
    本関数は将来的にCUDAに移植することを考えて
    forループを使って実装している。よってクソ遅い。
    耐えられないようであれば numba.jit のデコレータを使うこと。
    """

    # grid_num を算出しつつ lut の shape を確認
    # ------------------------------------------
    if (lut.shape[0] == lut.shape[1]) and (lut.shape[1] == lut.shape[2]):
        grid_num = np.uint8(lut.shape[0])
    else:
        print('lut format error. please set 4-dimentional lut.')
        return None

    img = img * (grid_num - 1)
    out_img = np.empty_like(img)
    slut = np.arange(grid_num + 1, dtype=np.uint32)
    slut[-1] = grid_num - 1  # あとでオーバーフロー回避で使う

    for w_idx in range(img.shape[1]):
        for h_idx in range(img.shape[0]):
            r = img[h_idx][w_idx][0]
            g = img[h_idx][w_idx][1]
            b = img[h_idx][w_idx][2]

            # r, g, b の各 Index を求める
            # ---------------------------
            r_idx = np.uint32(np.floor(r))
            g_idx = np.uint32(np.floor(g))
            b_idx = np.uint32(np.floor(b))

            # 体積算出のために r_0 ～ b_1 を求める
            # ------------------------------------
            r_0 = r - r_idx
            r_1 = 1 - r_0
            g_0 = g - g_idx
            g_1 = 1 - g_0
            b_0 = b - b_idx
            b_1 = 1 - b_0

            # 体積計算
            # -------------------------------------
            v_0 = (r_0 * g_0 * b_0)
            v_1 = (r_0 * g_0 * b_1)
            v_2 = (r_0 * g_1 * b_0)
            v_3 = (r_0 * g_1 * b_1)
            v_4 = (r_1 * g_0 * b_0)
            v_5 = (r_1 * g_0 * b_1)
            v_6 = (r_1 * g_1 * b_0)
            v_7 = (r_1 * g_1 * b_1)

            # l_0 ～ l_7 を事前に求めておく
            # --------------------------------------
            l_0 = lut[slut[r_idx+0]][slut[g_idx+0]][slut[b_idx+0]]
            l_1 = lut[slut[r_idx+0]][slut[g_idx+0]][slut[b_idx+1]]
            l_2 = lut[slut[r_idx+0]][slut[g_idx+1]][slut[b_idx+0]]
            l_3 = lut[slut[r_idx+0]][slut[g_idx+1]][slut[b_idx+1]]
            l_4 = lut[slut[r_idx+1]][slut[g_idx+0]][slut[b_idx+0]]
            l_5 = lut[slut[r_idx+1]][slut[g_idx+0]][slut[b_idx+1]]
            l_6 = lut[slut[r_idx+1]][slut[g_idx+1]][slut[b_idx+0]]
            l_7 = lut[slut[r_idx+1]][slut[g_idx+1]][slut[b_idx+1]]

            # 線形補間処理実行
            # -----------------------------------------
            r_out = v_0*l_7[0] + v_1*l_6[0] + v_2*l_5[0] + v_3*l_4[0]\
                + v_4*l_3[0] + v_5*l_2[0] + v_6*l_1[0] + v_7*l_0[0]
            g_out = v_0*l_7[1] + v_1*l_6[1] + v_2*l_5[1] + v_3*l_4[1]\
                + v_4*l_3[1] + v_5*l_2[1] + v_6*l_1[1] + v_7*l_0[1]
            b_out = v_0*l_7[2] + v_1*l_6[2] + v_2*l_5[2] + v_3*l_4[2]\
                + v_4*l_3[2] + v_5*l_2[2] + v_6*l_1[2] + v_7*l_0[2]

            # 結果を out_img に詰める
            # -----------------------------------------
            out_img[h_idx][w_idx][0] = r_out
            out_img[h_idx][w_idx][1] = g_out
            out_img[h_idx][w_idx][2] = b_out

    return out_img


def make_3dlut_data(grid_num=17, func=None, **kwargs):
    """
    # 概要
    3DLUTデータを作成する。

    # 詳細
    * func : 実際に処理をする関数を渡す
    * kwargs : func に引数として渡すやつ

    # 注意事項
    例によってエラー処理は皆無。トリッキーな呼び方はしないでね。
    """

    # 入力データ作成
    # -----------------
    x_r = (np.arange(grid_num**3) // (grid_num**0)) % grid_num
    x_g = (np.arange(grid_num**3) // (grid_num**1)) % grid_num
    x_b = (np.arange(grid_num**3) // (grid_num**2)) % grid_num

    x = (np.dstack((x_r, x_g, x_b)) / (grid_num - 1)).astype(np.float32)

    # LUTデータ作成
    # -----------------
    lut = func(in_data=x, **kwargs)

    # LUTデータは画像データと区別して、1次元配列っぽくする
    # ----------------------------------------------------
    if len(lut.shape) == 3:
        lut = lut.reshape((lut.shape[1], lut.shape[2]))

    return lut


def rgb2yuv_for_3dlut(in_data, **kwargs):
    """
    # 概要
    RGB2YUVの3DLUTデータを作る。

    # 引数
    kwargs['mtx'] に行列の係数を入れておくこと。
    以下は例。

    ```
    matrix_param = np.array([[0.2126, 0.7152, 0.0722],
                            [-0.114572, -0.385428, 0.5],
                            [0.5, -0.454153, -0.045847]])
    kwargs = {'mtx' : matrix_param}
    ```

    """

    if 'mtx' in kwargs:
        print('sonzai')
        mtx = kwargs['mtx']
    else:
        mtx = np.array([[0.2126, 0.7152, 0.0722],
                        [-0.114572, -0.385428, 0.5],
                        [0.5, -0.454153, -0.045847]])

    r_in, g_in, b_in = np.dsplit(in_data, 3)
    y = r_in * mtx[0][0] + g_in * mtx[0][1] + b_in * mtx[0][2]
    u = r_in * mtx[1][0] + g_in * mtx[1][1] + b_in * mtx[1][2] + 0.5
    v = r_in * mtx[2][0] + g_in * mtx[2][1] + b_in * mtx[2][2] + 0.5

    out_data = np.dstack((v, y, u))  # 並びが V, Y, U であることに注意

    return out_data


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


def sort_cube_data_to_4dim_array(in_lut):
    """
    # 概要
    [idx][3] の cube配列を
    [r_idx][g_idx][b_idx][3] の 4次元配列に並び替える
    """

    grid_num = np.uint8(np.round(np.power(in_lut.shape[0], 1/3)))
    out_lut = np.zeros((grid_num, grid_num, grid_num, 3), dtype=np.float32)

    for global_idx in range(grid_num**3):
        r_idx = (global_idx // (grid_num**0)) % grid_num
        g_idx = (global_idx // (grid_num**1)) % grid_num
        b_idx = (global_idx // (grid_num**2)) % grid_num
        out_lut[r_idx][g_idx][b_idx][0] = in_lut[global_idx][0]
        out_lut[r_idx][g_idx][b_idx][1] = in_lut[global_idx][1]
        out_lut[r_idx][g_idx][b_idx][2] = in_lut[global_idx][2]

    return out_lut


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


def check_time_3dlut(grid_num=17):
    # profiler関連の設定
    # ----------------------------------------
    config_file = "./data/config.txt"
    log_file = "./data/profile_out.csv"
    output_mode = cuda.profiler_output_mode.KEY_VALUE_PAIR
    cuda.initialize_profiler(config_file, log_file, output_mode)

    # 3DLUTデータの作成
    # ----------------------------------------
    matrix_param = np.array([[0.2126, 0.7152, 0.0722],
                            [-0.114572, -0.385428, 0.5],
                            [0.5, -0.454153, -0.045847]])
    kwargs = {'mtx': matrix_param}
    lut = make_3dlut_data(grid_num=17, func=rgb2yuv_for_3dlut, **kwargs)

    # 3DLUTの適用
    # ----------------------------------------
    img = img_open_and_normalize('../Matrix/figure/src_img.png')
    img_x86 = exec_3dlut_on_x86(img=img, lut=lut)
    cuda.start_profiler()
    img_gpu = exec_3dlut_on_gpu(img=img, lut=lut)
    cuda.stop_profiler()

    return img_x86, img_gpu


if __name__ == '__main__':
    if len(sys.argv) > 1:
        grid_num = sys.argv[1]
    else:
        print('grid_num = 17 is set')
        grid_num = 17

    check_time_3dlut(grid_num=grid_num)

