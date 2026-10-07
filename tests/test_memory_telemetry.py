import builtins
import errno
import tempfile
import unittest
from unittest.mock import patch

from robometer.utils.logger import log_memory_usage


class MemoryTelemetryTests(unittest.TestCase):
    def test_optional_memory_file_full_does_not_abort_training(self):
        with tempfile.TemporaryDirectory() as directory:
            original_open = builtins.open
            def mock_open(path, *args, **kwargs):
                if str(path).endswith('memory_log_rank0.txt'):
                    raise OSError(errno.ENOSPC, 'No space left on device')
                return original_open(path, *args, **kwargs)
            with patch('builtins.open', side_effect=mock_open):
                result = log_memory_usage('test', rank=0, output_dir=directory)
            self.assertIn('CPU:', result)


if __name__ == '__main__':
    unittest.main()
