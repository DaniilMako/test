import unittest
from birdCLEF import AudioAnalysisApp


class TestAudioAnalysisApp(unittest.TestCase):
    def setUp(self):
        self.app = AudioAnalysisApp()

    def test_flag_error(self):
        self.assertEqual(self.app.flag_error, False)


if __name__ == "__main__":
    unittest.main()
