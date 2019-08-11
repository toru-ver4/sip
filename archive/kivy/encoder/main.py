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
from kivy.resources import resource_add_path

Config.set('graphics', 'width', '800')
Config.set('graphics', 'height', '600')

resource_add_path('./font')


class GetFilePathDialog(FloatLayout):
    set_text = ObjectProperty(None)
    cancel = ObjectProperty(None)


class MainFrame(Widget):
    inout = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def open_get_file_path_dialog(self, inout):
        self.inout = inout
        content = GetFilePathDialog(set_text=self.set_text,
                                    cancel=self.dismiss_popup)
        self._popup = Popup(title="Select file", content=content,
                            size_hint=(0.95, 0.95))
        self._popup.open()

    def set_text(self, path, filename):
        filepath = os.path.join(path, filename[0])
        if self.inout == 'in':
            self.ids.text_input.text = filepath
            self.ids.text_output.text = filepath
            print(filepath)
        if self.inout == 'out':
            self.ids.text_output.text = filepath

        self.dismiss_popup()


class EncoderApp(App):
    def build(self):
        main_frame = MainFrame()
        return main_frame


if __name__ == '__main__':
    EncoderApp().run()
