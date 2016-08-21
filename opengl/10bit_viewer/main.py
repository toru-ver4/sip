import os
import sys
import time

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut


def display():
    pass


if __name__ == '__main__':
    glut.glutInit(sys.argv)
    glut.glutInitWindowSize(400, 400)
    glut.glutCreateWindow(b"30bit demo")
    glut.glutDisplayFunc(display)
    glut.glutMainLoop()

    # glut.glutInitDisplayString(b"red=10 green=10 blue=10 alpha=2")
    # glut.glutInitDisplayMode(glut.GLUT_RGB | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
