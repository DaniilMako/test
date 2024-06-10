import unittest
from unittest.mock import MagicMock
from birdCLEF import AudioAnalysisApp

class TestAudioAnalysisApp(unittest.TestCase):
    def setUp(self):
        self.app = AudioAnalysisApp()
        # Replace the method with MagicMock
        self.app.choose_audio_file = MagicMock(return_value="XC400498.ogg")

    def test_choose_audio_file(self):
        # Ensure that the method is being called
        self.app.choose_audio_file()
        # Assert that the method was called once
        self.app.choose_audio_file.assert_called_once()

    def test_load_model(self):
        # Replace the method with MagicMock
        self.app.load_model = MagicMock(return_value="mocked_model")
        model = self.app.load_model()
        # Assert that the mocked return value is returned
        self.assertEqual(model, "mocked_model")

    def test_mode_selection(self):
        # Assert the default mode
        self.assertEqual(self.app.mode.get(), "С тишиной")
        # Change mode and assert the new value
        self.app.mode.set("Без тишины")
        self.assertEqual(self.app.mode.get(), "Без тишины")

if __name__ == '__main__':
    unittest.main()
