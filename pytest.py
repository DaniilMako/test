import unittest
from birdCLEF import AudioAnalysisApp


class TestAudioAnalysisApp(unittest.TestCase):
    def setUp(self):
        self.app = AudioAnalysisApp()
        # self.app.audio_file_path = 'XC400498.ogg'

    def test_flag_error(self):
        self.app.start_analysis()
        self.assertEqual(self.app.flag_error, True)


if __name__ == "__main__":
    unittest.main()
