"""Проверки servo_sweep.py без Arduino.

python -m unittest discover -s tof_scan_tracker -p test_servo_sweep.py -v
"""
import contextlib
import io
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import servo_sweep as sweep
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


class ServoSweepTests(unittest.TestCase):
    def test_defaults_and_reject_invalid_arguments(self):
        args = sweep.parse_args([])
        self.assertEqual((args.port,args.x_min,args.x_max,args.y_min,args.y_max),
                         ('auto',40,140,55,125))
        for flags in (['--scan-speed','nan'], ['--servo-hz','0'],
                      ['--x-min','19'], ['--y-max','151'],
                      ['--x-min','80.1','--x-max','80.9']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): sweep.parse_args(flags)

    def test_snake_turns_at_both_vertical_edges(self):
        args=sweep.parse_args(['--x-min','80','--x-max','100','--y-min','80','--y-max','100'])
        path=sweep.SnakePath(args)
        rows=[]
        for _ in range(1000):
            previous=path.phase
            path.scan(.05)
            path.position_sent()
            self.assertTrue(80 <= path.x <= 100 and 80 <= path.y <= 100)
            if previous != 'SWEEP' and path.phase == 'SWEEP':
                rows.append((path.y,path.x_direction))
        self.assertEqual(rows[:5],[(80,1),(90,-1),(100,1),(90,-1),(80,1)])

    def test_corner_waits_for_command_before_changing_axis(self):
        path = sweep.SnakePath(sweep.parse_args([]))
        path.phase = 'SWEEP'
        path.x, path.y = 139,55
        path.scan(.1)
        self.assertEqual((path.x,path.y,path.pending_phase),(140,55,'ROW'))
        for _ in range(10):
            path.scan(.1)
        self.assertEqual((path.x,path.y),(140,55))
        path.position_sent()
        path.scan(.1)
        self.assertEqual((path.x,path.y),(140,57))

    def test_port_selection_and_ambiguity(self):
        uno=SimpleNamespace(device='/dev/ttyACM1',vid=0x2341,pid=0x0043)
        clone=SimpleNamespace(device='/dev/ttyUSB0',vid=0x1A86,pid=0x7523)
        self.assertEqual(sweep.resolve_port(ports=[uno,clone]),uno.device)
        self.assertEqual(sweep.resolve_port(ports=[clone]),clone.device)
        self.assertEqual(sweep.resolve_port('/dev/manual',ports=[]),'/dev/manual')
        with self.assertRaisesRegex(RuntimeError,'не найден'): sweep.resolve_port(ports=[])
        other=SimpleNamespace(device='/dev/ttyACM2',description='Arduino')
        with self.assertRaisesRegex(RuntimeError,'несколько'): sweep.resolve_port(ports=[uno,other])

    def test_pins_rate_and_fractional_limits(self):
        args=sweep.parse_args(['--port','/dev/test','--x-min','80.6','--x-max','100.4'])
        board=MagicMock()
        board.digital=[MagicMock() for _ in range(14)]
        with patch.object(sweep,'open_board',return_value=board), contextlib.redirect_stdout(io.StringIO()):
            rig=sweep.ServoRig(args)
        self.assertEqual([c.args[0] for c in board.servo_config.call_args_list],[9,10])
        rig.write(0,200,0)
        rig.write(90,90,.01)
        board.digital[9].write.assert_called_once_with(81)
        board.digital[10].write.assert_called_once_with(125)
        rig.write(200,0,.1)
        board.digital[9].write.assert_called_with(100)
        board.digital[10].write.assert_called_with(55)
        rig.write(200,0,.2)
        self.assertEqual(board.digital[9].write.call_count,2)

    def test_configuration_failure_closes_board(self):
        board=MagicMock()
        board.servo_config.side_effect=OSError(5,'config failed')
        with patch.object(sweep,'open_board',return_value=board), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError,'config failed'):
                sweep.ServoRig(sweep.parse_args(['--port','/dev/test']))
        board.exit.assert_called_once()

    def test_run_write_failure_closes_board_and_reports_pin(self):
        board=MagicMock()
        board.digital=[MagicMock() for _ in range(14)]
        board.digital[9].write.side_effect=OSError(5,'write failed')
        with patch.object(sweep,'open_board',return_value=board), patch.object(sweep,'sleep'), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'pin 9.*write failed'):
                sweep.run(sweep.parse_args(['--port','/dev/test']))
        board.exit.assert_called_once()

    def test_ctrl_c_during_positioning_closes_connection(self):
        rig=MagicMock()
        with patch.object(sweep,'ServoRig',return_value=rig), \
                patch.object(sweep,'sleep',side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt): sweep.run(sweep.parse_args([]))
        rig.close.assert_called_once()

    def test_dry_run_does_not_open_board(self):
        with patch.object(sweep,'open_board') as connection:
            rig=sweep.ServoRig(sweep.parse_args(['--dry-run']))
            rig.write(90,90,0)
            rig.close()
        connection.assert_not_called()


if __name__=='__main__':
    unittest.main()
