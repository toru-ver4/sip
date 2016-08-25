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


def resize(w, h):
    pass
    # gl.glViewport(0, 0, w, h)
    # gl.glMatrixMode(gl.GL_PROJECTION)
    # gl.glLoadIdentity()
    # gl.glMatrixMode(gl.GL_MODELVIEW)

    # if False:
    #     gl.glOrtho(-w/const_window_width, w/const_window_width,
    #                -h/const_window_height, h/const_window_height,
    #                -1.0, 1.0)
    # スクリーン上の座標系をマウスの座標系に一致させる
    # gl.glOrtho(-0.5, w - 0.5,
    #            h - 0.5, -0.5,
    #            -1.0, 1.0)


def init():
    global texture
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    img = get_img()

    img = np.reshape(img, (img.shape[1], img.shape[0], img.shape[2]))
    print(img.shape)
    img = np.float32(img/0xFFFF)
    print(img.dtype)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F,
                    img.shape[0], img.shape[1], 0, gl.GL_RGB,
                    gl.GL_FLOAT, img.tostring())
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)


def get_img():
    img = cv2.imread('./figure/10bit_src.tiff',
                     cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    # print(img)
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
    glut.glutInit(sys.argv)
    glut.glutInitWindowSize(const_window_width, const_window_height)
    # glut.glutInitDisplayString(b"red=10 green=10 blue=10 alpha=2")
    # glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DEPTH)
    glut.glutInitDisplayMode(glut.GLUT_RGBA)
    glut.glutCreateWindow(b"30bit demo")
    glut.glutDisplayFunc(display)
    glut.glutReshapeFunc(resize)
    init()
    glut.glutMainLoop()
