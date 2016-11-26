#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
日本語Kivy解説書でHello World的な
"""
 
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle


class Field(Widget):
    def __init__(self):
        super(Field, self).__init__()
        self.canvas.add(Rectangle(source='background.jpg', size=(1024,768)))
        bird = Bird()
        self.add_widget(bird)


class Bird(Widget):
    def __init__(self):
        super(Bird, self).__init__()
        self.canvas.add(Rectangle(source='unko.png', 
                                  size=(100,200), pos=(100,100)))
    


class MyApp(App):
    def build(self):
        return Field()


if __name__ == '__main__':
    MyApp().run()
