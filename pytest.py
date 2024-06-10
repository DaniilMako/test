import unittest
from unittest.mock import patch
from birdCLEF import AudioAnalysisApp


class TestAudioAnalysisApp(unittest.TestCase):

    @patch('birdCLEF.filedialog.askopenfilename', return_value='XC400498.ogg')
    @patch('tkinter.Tk.__init__', return_value=None)
    def test_choose_audio_file(self, mock_tk, mock_askopenfilename):
        app = AudioAnalysisApp()
        app.choose_audio_file()
        self.assertEqual(app.audio_file_path, 'XC400498.ogg')
        self.assertTrue(mock_askopenfilename.called)

    @patch('tkinter.Tk.__init__', return_value=None)
    def test_create_widgets(self, mock_tk):
        app = AudioAnalysisApp()
        app.create_widgets()
        self.assertTrue(app.result_tabs.winfo_exists())


if __name__ == '__main__':
    unittest.main()
