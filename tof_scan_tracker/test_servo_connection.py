"""Проверка ошибок USB без подключения Arduino."""
import contextlib
import io
import unittest
from servo_sweep import open_board


class Serial:
    def __init__(self): self.closed=False
    def close(self): self.closed=True


class ConnectionTests(unittest.TestCase):
    def test_exit_once_even_with_destructor(self):
        class Board:
            def __init__(self, port):
                self.sp=Serial()
                self.exits=0
            def exit(self): self.exits+=1
        board=open_board('/dev/test',Board)
        board.exit()
        board.exit()
        board.__del__()
        self.assertEqual(board.exits,1)
        self.assertTrue(board.sp.closed)

    def test_disconnect_during_exit_still_closes_serial(self):
        class Board:
            def __init__(self, port): self.sp=Serial()
            def exit(self): raise OSError(5,'Input/output error')
        board=open_board('/dev/test',Board)
        with contextlib.redirect_stderr(io.StringIO()) as log:
            board.exit()
            board.__del__()
        self.assertTrue(board.sp.closed)
        self.assertEqual(log.getvalue().count('Input/output error'),1)

    def test_partial_initialization_preserves_original_error(self):
        serial=Serial()
        class Board:
            def __init__(self,port):
                self.sp=serial
                raise OSError(5,'initial write failed')
            def exit(self): raise AssertionError('Not initialized')
        with self.assertRaisesRegex(RuntimeError,'initial write failed'):
            open_board('/dev/test',Board)
        self.assertTrue(serial.closed)

    def test_original_write_error_not_masked_by_cleanup(self):
        class Board:
            def __init__(self,port): self.sp=Serial()
            def exit(self): raise OSError('cleanup error')
        board=open_board('/dev/test',Board)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(OSError,'original write'):
                try:
                    raise OSError('original write')
                finally:
                    board.exit()
        self.assertTrue(board.sp.closed)


if __name__=='__main__':
    unittest.main()
