import unittest
from unittest.mock import patch
from birdCLEF import AudioAnalysisApp


class TestAudioAnalysisApp(unittest.TestCase):
    def test_create_widgets(self):
        # root = Tk()
        app = AudioAnalysisApp()
        app.create_widgets()
        # You can add more assertions here to check if widgets are created as expected
        self.assertTrue(app.result_tabs.winfo_exists())  # Assert that result_tabs exists

    @patch('birdCLEF.filedialog.askopenfilename', return_value='XC400498.ogg')
    def test_choose_audio_file(self, mock_askopenfilename):
        # root = Tk()
        app = AudioAnalysisApp()
        app.choose_audio_file()
        self.assertEqual(app.audio_file_path, 'XC400498.ogg')
        self.assertTrue(mock_askopenfilename.called)


if __name__ == '__main__':
    unittest.main()
