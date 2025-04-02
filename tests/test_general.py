import platform
import unittest
import shutil


class TestGeneral(unittest.TestCase):
    """Tests general setup things, such as pre-requisites."""

    def test_os(self):
        """Checks if the system is a Unix system."""
        self.assertIn(
            platform.system(),
            ["Darwin", "Linux"],
            msg="This code was not developed on Windows and as such may not work as well.",
        )

    def test_ollama(self):
        """Tests if the system has ollama installed."""
        self.assertTrue(
            shutil.which("ollama"),
            "Ollama is not installed on this computer. Please install ollama using your system's package manager or from https://ollama.com/download. ",
        )

    def test_conda(self):
        """Tests if the system has conda installed."""
        self.assertTrue(
            shutil.which("conda"),
            "Conda is not installed on this computer. Please install conda using your package manager or from https://www.anaconda.com/download. ",
        )


if __name__ == "__main__":
    unittest.main()
