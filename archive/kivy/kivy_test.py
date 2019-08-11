#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
kivyのHello World + alpha.
"""
 
from kivy.app import App
from kivy.uix.button import Button

class TestApp(App):
    def build(self):
        return Button(text='Hello World')

if __name__ == '__main__':
    TestApp().run()
    
