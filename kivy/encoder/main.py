#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""
kivyのHello World + alpha.
"""

import os
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.config import Config
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup

Config.set('graphics', 'width', '400')
Config.set('graphics', 'height', '600')


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    cancel = ObjectProperty(None)


class MainFrame(Widget):
    # loadfile = ObjectProperty(None)
    # savefile = ObjectProperty(None)
    # text_input = ObjectProperty(None)

    def get_input_file(self, dirpath="C:/Users/toruv/Downloads"):

        print("selected")

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.95, 0.95))
        self._popup.open()

    def load(self, path, filename):
        self.ids.text_input.text = os.path.join(path, filename[0])
        self.dismiss_popup()

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.95, 0.95))
        self._popup.open()

    def save(self, path, filename):
        self.ids.text_output.text = os.path.join(path, filename[0])
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()


class EncoderApp(App):
    def build(self):
        main_frame = MainFrame()
        return main_frame


if __name__ == '__main__':
    EncoderApp().run()
