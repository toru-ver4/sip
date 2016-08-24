import os
import sys
import time
from PIL import Image
import cv2

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

const_window_width = 2048
const_window_height = 1024
global_x0 = 0
global_y0 = 0
texture = None


def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glBegin(gl.GL_POLYGON)
    gl.glColor3d(1.0, 0.0, 0.0)
    gl.glVertex2d(-0.9, -0.9)
    gl.glVertex2d(0.9, -0.9)
    gl.glVertex2d(0.9, 0.9)
    gl.glVertex2d(-0.9, 0.9)
    gl.glEnd()
    gl.glFlush()


def resize(w, h):
    gl.glViewport(0, 0, w, h)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glMatrixMode(gl.GL_MODELVIEW)

    # if False:
    #     gl.glOrtho(-w/const_window_width, w/const_window_width,
    #                -h/const_window_height, h/const_window_height,
    #                -1.0, 1.0)
    # スクリーン上の座標系をマウスの座標系に一致させる
    # gl.glOrtho(-0.5, w - 0.5,
    #            h - 0.5, -0.5,
    #            -1.0, 1.0)


# def mouse(button, state, x, y):
#     global global_x0
#     global global_y0

#     if button == glut.GLUT_LEFT_BUTTON:
#         if state == glut.GLUT_UP:
#             gl.glColor3d(0.0, 0.0, 0.0)
#             gl.glBegin(gl.GL_LINES)
#             gl.glVertex2i(global_x0, global_y0)
#             gl.glVertex2i(x, y)
#             gl.glEnd()
#             gl.glFlush()
#             print("pass")
#         else:
#             global_x0 = x
#             global_y0 = y

#     print("x={}, y={}, x0={}, y0={}".format(x, y, global_x0, global_y0))


def init():
    global texture
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    gl.glDepthFunc(gl.GL_LEQUAL)
    # gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    img = get_img()

    # alpha channel データを作成＆結合！
    # ----------------------------------
    alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) * 0xFF
    r, g, b = cv2.split(img)
    img = cv2.merge((r, g, b, alpha))
    img = np.reshape(img, (img.shape[1], img.shape[0], img.shape[2]))
    print(img.shape)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8UI,
                    img.shape[0], img.shape[1], 0, gl.GL_RGBA_INTEGER,
                    gl.GL_UNSIGNED_BYTE, img.tostring())


def get_img():
    img = cv2.imread('./figure/10bit_src.tiff',
                     cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    img = img >> 8
    img = np.uint8(img)
    # print(img)
    return img


def drawQuads():
    global texture
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    # gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glColor3f(0.0, 1.0, 0.0);
    gl.glBegin(gl.GL_QUADS)
    gl.glTexCoord2f(0.0, 1.0)
    gl.glVertex2f(-1.0, 1.0)  # Bottom Left
    gl.glTexCoord2f(0.0, 0.0)
    gl.glVertex2f(-1.0, -1.0)  # Top Left
    gl.glTexCoord2f(1.0, 0.0)
    gl.glVertex2f(1.0, -1.0)  # Top Right
    gl.glTexCoord2f(1.0, 1.0)
    gl.glVertex2f(1.0, 1.0)  # Bottom Right
    gl.glEnd()
    # gl.glFinish()
    gl.glFlush()


if __name__ == '__main__':
    glut.glutInit(sys.argv)
    glut.glutInitWindowSize(const_window_width, const_window_height)
    # glut.glutInitDisplayString(b"red=10 green=10 blue=10 alpha=2")
    glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DEPTH)
    # glut.glutInitDisplayMode(glut.GLUT_RGBA)
    glut.glutCreateWindow(b"30bit demo")
    glut.glutDisplayFunc(drawQuads)
    # glut.glutReshapeFunc(resize)
    # glut.glutMouseFunc(mouse)
    init()
    glut.glutMainLoop()
