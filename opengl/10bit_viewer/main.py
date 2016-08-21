import os
import sys
import time

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

const_window_width = 400
const_window_height = 800


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

    gl.glOrtho(-w/const_window_width, w/const_window_width,
               -h/const_window_height, h/const_window_height,
               -1.0, 1.0)


def init():
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)


if __name__ == '__main__':
    glut.glutInit(sys.argv)
    glut.glutInitWindowSize(const_window_width, const_window_height)
    glut.glutInitDisplayMode(glut.GLUT_RGBA)
    glut.glutCreateWindow(b"30bit demo")
    glut.glutDisplayFunc(display)
    glut.glutReshapeFunc(resize)
    init()
    glut.glutMainLoop()

    # glut.glutInitDisplayString(b"red=10 green=10 blue=10 alpha=2")
    # glut.glutInitDisplayMode(glut.GLUT_RGB | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
