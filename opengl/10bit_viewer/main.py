import os
import sys
import time

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut


def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glColor3d(1.0, 0.0, 0.0)
    gl.glBegin(gl.GL_LINE_LOOP)
    gl.glVertex2d(-0.9, -0.9)
    gl.glVertex2d(0.9, -0.9)
    gl.glVertex2d(0.9, 0.9)
    gl.glVertex2d(-0.9, 0.9)
    gl.glEnd()
    gl.glFlush()


def init():
    gl.glClearColor(0.0, 0.0, 1.0, 1.0)


if __name__ == '__main__':
    glut.glutInit(sys.argv)
    glut.glutInitWindowSize(400, 400)
    glut.glutInitDisplayMode(glut.GLUT_RGBA)
    glut.glutCreateWindow(b"30bit demo")
    glut.glutDisplayFunc(display)
    init()
    glut.glutMainLoop()

    # glut.glutInitDisplayString(b"red=10 green=10 blue=10 alpha=2")
    # glut.glutInitDisplayMode(glut.GLUT_RGB | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
