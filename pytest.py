import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk
import matplotlib


matplotlib.use('Agg')

from birdCLEF import AudioAnalysisApp

class TestAudioAnalysisApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock для tkinter.Tk
        cls.root = MagicMock()
        cls.tk_patcher = patch('tkinter.Tk', return_value=cls.root)
        cls.tk_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.tk_patcher.stop()

    def test_create_widgets(self):
        app = AudioAnalysisApp()
        app.create_widgets()
        self.assertTrue(app.result_tabs.winfo_exists())

    @patch('birdCLEF.filedialog.askopenfilename', return_value='XC400498.ogg')
    def test_choose_audio_file(self, mock_askopenfilename):
        app = AudioAnalysisApp()
        app.choose_audio_file()
        self.assertEqual(app.audio_file_path, 'XC400498.ogg')
        self.assertTrue(mock_askopenfilename.called)

if __name__ == '__main__':
    unittest.main()
