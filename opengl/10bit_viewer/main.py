import os
import sys
import time

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

const_window_width = 2048
const_window_height = 1024
global_x0 = 0
global_y0 = 0


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
    gl.glLoadIdentity()

    # if False:
    #     gl.glOrtho(-w/const_window_width, w/const_window_width,
    #                -h/const_window_height, h/const_window_height,
    #                -1.0, 1.0)
    # スクリーン上の座標系をマウスの座標系に一致させる
    gl.glOrtho(-0.5, w - 0.5,
               h - 0.5, -0.5,
               -1.0, 1.0)


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
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 2)
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
    gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
    gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    img = get_img()
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB16UI,
                    img.shape[0], img.shape[1], 0, gl.GL_RGB_INTEGER,
                    gl.GL_UNSIGNED_SHORT, img)


def get_img():
    x = np.arange(const_window_width*const_window_height * 3,
                  dtype=np.uint32) % 1024
    x = x * 64
    img = np.uint16(np.reshape(x, (const_window_width,
                                   const_window_height, 3)))
    return img


def drawQuads():

    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBegin(gl.GL_QUADS)
    gl.glColor3f(0.0, 0.0, 0.0);
    gl.glVertex2f(-1.0, -1.0)
    gl.glColor3f(0.0, 0.0, 0.0)
    gl.glVertex2f(-1.0, 1.0)
    gl.glColor3f(0.0, 1.0, 0.0)
    gl.glVertex2f(1.0, 1.0)
    gl.glColor3f(0.0, 1.0, 0.0)
    gl.glVertex2f(1.0, -1.0)
    gl.glEnd()
    gl.glFlush()


if __name__ == '__main__':
    glut.glutInit(sys.argv)
    glut.glutInitWindowSize(const_window_width, const_window_height)
    glut.glutInitDisplayString(b"red=10 green=10 blue=10 alpha=2")
    glut.glutInitDisplayMode(glut.GLUT_RGB | glut.GLUT_DEPTH)
    # glut.glutInitDisplayMode(glut.GLUT_RGBA)
    glut.glutCreateWindow(b"30bit demo")
    glut.glutDisplayFunc(drawQuads)
    # glut.glutReshapeFunc(resize)
    # glut.glutMouseFunc(mouse)
    init()
    glut.glutMainLoop()

