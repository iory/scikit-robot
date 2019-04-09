import unittest

from skrobot.utils.download import cached_gdown_download


class TestDownload(unittest.TestCase):

    def test_cached_gdown_download(self):
        cached_gdown_download(
            'https://drive.google.com/uc?id=1j_9MF3Yftk1G6--5W2zUnRxb9k9qfx7z',
            md5sum='d41d8cd98f00b204e9800998ecf8427e')
        cached_gdown_download(
            'https://drive.google.com/uc?id=1j_9MF3Yftk1G6--5W2zUnRxb9k9qfx7z',
            md5sum='d41d8cd98f00b204e9800998ecf8427e')
