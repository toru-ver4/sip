import os
import sys
import time
from PIL import Image
import cv2
import ctypes
import six
import packaging
import packaging.version
import packaging.specifiers
import packaging.requirements

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

g_window_width = 3840
g_window_height = 2160
global_x0 = 0
global_y0 = 0
texture = None


def resize(w, h):
    pass
    # gl.glViewport(0, 0, w, h)
    # gl.glMatrixMode(gl.GL_PROJECTION)
    # gl.glLoadIdentity()
    # gl.glMatrixMode(gl.GL_MODELVIEW)

    # if False:
    #     gl.glOrtho(-w/g_window_width, w/g_window_width,
    #                -h/g_window_height, h/g_window_height,
    #                -1.0, 1.0)
    # スクリーン上の座標系をマウスの座標系に一致させる
    # gl.glOrtho(-0.5, w - 0.5,
    #            h - 0.5, -0.5,
    #            -1.0, 1.0)


def keyboard(key, x, y):
    if (key == b'q') or (key == b'\x1b'):
        sys.exit()


def get_winodw_resolution():
    user32 = ctypes.windll.user32
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)

    return width, height


def init():
    global texture
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    img = get_img()

    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F,
                    img.shape[0], img.shape[1], 0, gl.GL_RGB,
                    gl.GL_FLOAT, img.tostring())
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)


def get_img():
    filename = './figure/10bit_src.tiff'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = './figure/10bit_src.tiff'
    img = cv2.imread(filename,
                     cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    # フルスクリーン用に黒枠作成 ＆ 合成
    # ------------------------------------
    color_num = img.shape[2]

    h_offset = int((g_window_width - img.shape[1]) / 2)
    v_offset = int((g_window_height - img.shape[0]) / 2)

    if (h_offset < 0) or (v_offset < 0):
        print("===================================================")
        print("==== Caution! Picture Size is too Big. ============")
        print("===================================================")
        h_offset = 0
        v_offset = 0
        img = img[0:g_window_height, 0:g_window_width, :]
    full_img = np.zeros((g_window_height, g_window_width, color_num),
                        dtype=img.dtype)
    full_img[v_offset:(img.shape[0] + v_offset),
             h_offset:img.shape[1] + h_offset, :] = img
    img = full_img

    # OpenCV は V --> H の並びなので、reshape して H --> V にする
    # ------------------------------------
    img = np.reshape(img, (img.shape[1], img.shape[0], img.shape[2]))
    print(img.shape)
    print(img.dtype)

    # dtype から型の最大値を求めて正規化する
    # ------------------------------------
    try:
        img_max_value = np.iinfo(img.dtype).max
    except:
        img_max_value = 1.0
    print(img_max_value)

    img = np.float32(img/img_max_value)

    # cv2.imshow('preview', full_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img[:, :, ::-1]


def display():
    global texture
    # gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glColor3f(1.0, 1.0, 1.0)
    gl.glBegin(gl.GL_QUADS)
    gl.glTexCoord2f(0.0, 1.0)
    gl.glVertex3d(-1.0, -1.0, 0.0)  # Bottom Left
    gl.glTexCoord2f(1.0, 1.0)
    gl.glVertex3d(1.0, -1.0, 0.0)  # Top Left
    gl.glTexCoord2f(1.0, 0.0)
    gl.glVertex3d(1.0, 1.0, 0.0)  # Top Right
    gl.glTexCoord2f(0.0, 0.0)
    gl.glVertex3d(-1.0, 1.0, 0.0)  # Bottom Right
    gl.glEnd()
    gl.glDisable(gl.GL_TEXTURE_2D)
    # gl.glFinish()
    gl.glFlush()


if __name__ == '__main__':
    game_mode = True
    glut.glutInit(sys.argv)
    g_window_width, g_window_height = get_winodw_resolution()
    glut.glutInitWindowSize(g_window_width, g_window_height)
    glut.glutInitDisplayString(b"red=10 green=10 blue=10 alpha=2")
    # glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DEPTH)
    glut.glutInitDisplayMode(glut.GLUT_RGBA)
    if game_mode:
        mode_string = "{}x{}:32@60".format(g_window_width, g_window_height)
        glut.glutGameModeString(mode_string)
        glut.glutEnterGameMode()
    else:
        glut.glutCreateWindow(b"30bit demo")
    glut.glutDisplayFunc(display)
    glut.glutReshapeFunc(resize)
    glut.glutKeyboardFunc(keyboard)
    init()
    glut.glutMainLoop()
