import unittest
from unittest.mock import MagicMock
from birdCLEF import AudioAnalysisApp

class TestAudioAnalysisApp(unittest.TestCase):
    def setUp(self):
        self.app = AudioAnalysisApp()
        self.app.choose_audio_file = MagicMock(return_value="XC400498.ogg")

    def test_choose_audio_file(self):
        self.app.choose_audio_file()
        self.app.choose_audio_file.assert_called_once()

    def test_load_model(self):
        self.app.load_model = MagicMock(return_value="mocked_model")
        model = self.app.load_model()
        self.assertEqual(model, "mocked_model")

    def test_mode_selection(self):
        self.assertEqual(self.app.mode.get(), "С тишиной")
        self.app.mode.set("Без тишины")
        self.assertEqual(self.app.mode.get(), "Без тишины")

if __name__ == '__main__':
    unittest.main()
