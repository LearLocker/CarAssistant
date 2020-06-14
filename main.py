# coding:utf-8
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.config import ConfigParser

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.gridlayout import GridLayout
import cv2
import os
import ast
import time
from RoadLineDetect import RoadLineDetect
from RoadSignDetect import RoadSignDetect

class MainScreen(Screen):
    def __init__(self, **kw):
        super(MainScreen, self).__init__(**kw)

        ModeRoadButton = Button(text='Детектирование дорожной разметки',
                                on_press=lambda x:
                                set_screen('RoadLine'),
                                font_size=14);

        ModeSignButton = Button(text='Детектирование дорожных знаков',
                                on_press=lambda x: set_screen('RoadSign'),
                                font_size = 14);

        MainLayout = AnchorLayout()
        ControlLayout = BoxLayout(orientation='vertical', size_hint=[.7,.9], padding = 100, spacing = 10)
        ControlLayout.add_widget(Label(text="Выберите режим работы приложения", font_size=14, size_hint=[1, .2]))
        ControlLayout.add_widget(Widget())
        ControlLayout.add_widget(ModeRoadButton)
        ControlLayout.add_widget(ModeSignButton)
        MainLayout.add_widget(ControlLayout)
        self.add_widget(MainLayout)


def set_screen(name_screen):
    screenManager.current = name_screen

screenManager = ScreenManager()
screenManager.add_widget(MainScreen(name='Main'))
screenManager.add_widget(RoadLineDetect.RoadLineDetection(name='RoadLine'))
screenManager.add_widget(RoadSignDetect.RoadSignDetection(name='RoadSign'))


class HelpDrivingApp(App):
    def __init__(self, **kvargs):
        super(HelpDrivingApp, self).__init__(**kvargs)
        self.config = ConfigParser()

    def build_config(self, config):
        config.adddefaultsection('General')
        config.setdefault('General', 'user_data', '{}')

    def set_value_from_config(self):
        self.config.read(os.path.join(self.directory, '%(appname)s.ini'))
        self.user_data = ast.literal_eval(self.config.get(
            'General', 'user_data'))

    def get_application_config(self):
        return super(HelpDrivingApp, self).get_application_config(
            '{}/%(appname)s.ini'.format(self.directory))

    def build(self):
        return screenManager

if __name__ == '__main__':
    HelpDrivingApp().run()