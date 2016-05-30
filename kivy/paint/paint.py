#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
paint application.
"""

import numpy.random

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse


from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line


class MyPaintWidget(Widget):

    seed = 0
    def on_touch_down(self, touch):
        with self.canvas:
            numpy.random.seed(self.seed)
            self.seed += 1
            h_param = numpy.random.random(1) 
            Color(h_param, 1, 1, mode='hsv')
            d = 30.
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class MyPaintApp(App):

    def build(self):
        return MyPaintWidget()


if __name__ == '__main__':
    MyPaintApp().run()
    