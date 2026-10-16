"""python -m unittest discover -s tof_scan_tracker -p test_scan_track.py -v"""
import unittest
from unittest.mock import patch
import contextlib
import io
from types import SimpleNamespace
import numpy as np
from scan_track import parse_args, ScanTracker, ServoRig, find_fragments
import scan_track
import servo_sweep


def target(x=.5, y=.5, distance=450, area=100):
    return dict(center=(x, y), distance=distance, area=area, confidence=300)


class TrackerTests(unittest.TestCase):
    def test_defaults(self):
        a = parse_args([])
        self.assertEqual((a.min, a.max, a.confidence, a.lost_timeout), (400, 500, 200, 3))
        self.assertEqual(a.port, 'auto')
        self.assertFalse(a.scan_only)
        self.assertTrue(a.invert_x)
        self.assertFalse(a.invert_y)

    def test_scan_only_ignores_visible_targets(self):
        t = ScanTracker(parse_args(['--scan-only']))
        for i in range(100):
            self.assertIsNone(t.update([target()], i*.05))
            self.assertEqual(t.state, 'SCAN')
        self.assertNotEqual((t.x,t.y), (90,90))
        self.assertIsNone(t.last_seen)

    def test_port_detection(self):
        def port(device, vid=None, pid=None, description=''):
            return SimpleNamespace(device=device,vid=vid,pid=pid,description=description)
        uno = port('/dev/ttyACM1', 0x2341, 0x0043)
        clone = port('/dev/ttyUSB0', 0x1A86, 0x7523)
        unrelated = port('/dev/ttyS0', description='UART')
        self.assertEqual(scan_track.resolve_port(ports=[uno,clone,unrelated]),uno.device)
        self.assertEqual(scan_track.resolve_port(ports=[clone,unrelated]),clone.device)
        self.assertEqual(scan_track.resolve_port('/dev/custom',ports=[]),'/dev/custom')
        with self.assertRaisesRegex(RuntimeError,'не найден'):
            scan_track.resolve_port(ports=[unrelated])
        with self.assertRaisesRegex(RuntimeError,'несколько'):
            scan_track.resolve_port(ports=[uno,port('/dev/ttyACM2',description='Arduino Uno')])

    def test_wait_reacquire_and_resume(self):
        t = ScanTracker(parse_args([]))
        t.update([target()], 0)
        self.assertEqual(t.state, 'TRACK')
        held = t.x, t.y
        t.update([], 1)
        t.update([], 2.99)
        self.assertEqual(t.state, 'WAIT')
        self.assertEqual((t.x, t.y), held)
        t.update([target()], 2.999)
        self.assertEqual(t.state, 'TRACK')
        t.update([], 5.998)
        self.assertEqual(t.state, 'WAIT')
        t.update([], 6)
        self.assertEqual(t.state, 'SCAN')

    def test_no_switch_to_unrelated_nearer_object(self):
        t = ScanTracker(parse_args([]))
        t.update([target(.2, .5)], 0)
        selected = t.update([target(.21, .5), target(.9, .5, 410)], .1)
        self.assertEqual(selected['center'], (.21, .5))
        self.assertIsNone(t.update([target(.9, .5, 410)], .2))
        self.assertEqual(t.state, 'WAIT')

    def test_tracking_direction_inversion_and_deadzone(self):
        for flags, sign_x, sign_y in (([], -1, 1),
                                      (['--no-invert-x', '--invert-y'], 1, -1)):
            t = ScanTracker(parse_args(flags))
            t.update([target(.8,.8)], 0)
            t.update([target(.8,.8)], .1)
            self.assertGreater(sign_x*(t.x-90), 0)
            self.assertGreater(sign_y*(t.y-90), 0)
        t = ScanTracker(parse_args([]))
        t.update([target(.52,.48)], 0)
        t.update([target(.52,.48)], .1)
        self.assertEqual((t.x,t.y), (90,90))

    def test_default_x_direction_for_target_on_left(self):
        t = ScanTracker(parse_args([]))
        t.update([target(.2, .5)], 0)
        t.update([target(.2, .5)], .1)
        self.assertGreater(t.x, 90)
        self.assertEqual(t.y, 90)

    def test_snake_rows_reverse_and_angles_bounded(self):
        a = parse_args(['--x-min','80','--x-max','100','--y-min','80','--y-max','100'])
        t = ScanTracker(a)
        rows = []
        for i in range(1000):
            old_phase = t.phase
            t.update([], i*.05)
            self.assertTrue(80 <= t.x <= 100 and 80 <= t.y <= 100)
            if old_phase != 'SWEEP' and t.phase == 'SWEEP':
                rows.append((t.y,t.x_direction))
        self.assertEqual(rows[:5], [(80,1), (90,-1), (100,1), (90,-1), (80,1)])

    def test_long_pause_does_not_jump(self):
        t = ScanTracker(parse_args([]))
        t.update([target(1,1)], 0)
        t.update([target(1,1)], 100)
        self.assertLessEqual(abs(t.x-90), 4)
        self.assertLessEqual(abs(t.y-90), 4)

    def test_fragment_filter_and_centroid(self):
        d = np.zeros((10,10), np.float32)
        d[2:6,3:7] = 450
        c = np.full(d.shape,300,np.float32)
        a = parse_args(['--min-area','4'])
        f = find_fragments(d,c,a)
        self.assertEqual(len(f),1)
        self.assertEqual(f[0]['area'],16)
        self.assertAlmostEqual(f[0]['center'][0],.5)
        self.assertAlmostEqual(f[0]['center'][1],3.5/9)
        d[2:6,3:7] = 501
        self.assertEqual(find_fragments(d,c,a),[])

    def test_servo_pins_limits_and_command_rate(self):
        class Pin:
            def __init__(self): self.values=[]
            def write(self,v): self.values.append(v)
        class Board:
            def __init__(self):
                self.digital=[Pin() for _ in range(14)]
                self.config=[]
                self.closed=False
            def servo_config(self,pin,angle): self.config.append((pin,angle))
            def exit(self): self.closed=True
        board=Board()
        with patch.object(servo_sweep, 'open_board', return_value=board):
            rig=ServoRig(parse_args(['--port','/dev/test']))
        self.assertEqual(board.config,[(9,90),(10,90)])
        rig.write(-100,500,0)
        rig.write(90,90,.01)
        self.assertEqual(board.digital[9].values,[20])
        self.assertEqual(board.digital[10].values,[150])
        rig.write(90,90,.1)
        self.assertEqual(board.digital[9].values,[20,90])
        rig.close()
        self.assertTrue(board.closed)

    def test_capture_error_releases_camera_and_arduino(self):
        from contextlib import ExitStack
        class Frame:
            depth_data = np.full((12,12),450,np.float32)
            confidence_data = None
        class Camera:
            released = stopped = closed = False
            def open(self,*_): return 0
            def start(self,*_): return 0
            def setControl(self,*_): return 0
            def getControl(self,*_): return 4000
            def requestFrame(self,*_): return Frame()
            def releaseFrame(self,*_): self.released=True
            def stop(self): self.stopped=True
            def close(self): self.closed=True
        camera=Camera()
        sdk=SimpleNamespace(ArducamCamera=lambda:camera, DepthData=Frame,
            Connection=SimpleNamespace(CSI=0),FrameType=SimpleNamespace(DEPTH=0),
            Control=SimpleNamespace(RANGE=0))
        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules',{'ArducamDepthCamera':sdk}))
            rig=stack.enter_context(patch.object(scan_track,'ServoRig')).return_value
            for name in ('namedWindow','resizeWindow','imshow','destroyAllWindows'):
                stack.enter_context(patch.object(scan_track.cv2,name))
            with self.assertRaisesRegex(RuntimeError,'depth/confidence'):
                scan_track.run(parse_args([]))
            rig.close.assert_called_once()
            rig.write.assert_not_called()
        self.assertTrue(camera.released and camera.stopped and camera.closed)

    def test_scan_matches_servo_sweep_trajectory(self):
        args=parse_args(['--scan-only','--x-min','80','--x-max','100',
                         '--y-min','80','--y-max','100'])
        tracker=ScanTracker(args)
        path=servo_sweep.SnakePath(args)
        tracker.update([],0)
        for i in range(1,1000):
            path.scan(.05)
            tracker.update([target()],i*.05)
            self.assertAlmostEqual(tracker.x,path.x,places=6)
            self.assertAlmostEqual(tracker.y,path.y,places=6)

    def test_rejects_sector_without_integer_angle(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--x-min','80.1','--x-max','80.9'])

    def test_tracking_never_exceeds_fractional_limits(self):
        args=parse_args(['--x-min','80.6','--x-max','90.4',
                         '--y-min','80.6','--y-max','90.4','--no-servos'])
        tracker=ScanTracker(args)
        rig=ServoRig(args)
        for i in range(200):
            tracker.update([target(1,1)], i*.1)
            rig.write(tracker.x,tracker.y,i*.1)
            self.assertTrue(all(80.6 <= angle <= 90.4 for angle in rig.last_command))

    def test_usb_write_failure_keeps_original_error_and_releases_resources(self):
        from contextlib import ExitStack
        class Frame:
            depth_data=np.full((12,12),450,np.float32)
            confidence_data=np.full((12,12),300,np.float32)
        class Camera:
            released=stopped=closed=False
            def open(self,*_): return 0
            def start(self,*_): return 0
            def setControl(self,*_): return 0
            def getControl(self,*_): return 4000
            def requestFrame(self,*_): return Frame()
            def releaseFrame(self,*_): self.released=True
            def stop(self): self.stopped=True
            def close(self): self.closed=True
        class Serial:
            closed=False
            def close(self): self.closed=True
        instances=[]
        class Pin:
            def write(self,angle): raise OSError(5,'original write failed')
        class Board:
            def __init__(self,port):
                self.sp=Serial()
                self.digital=[Pin() for _ in range(14)]
                self.exits=0
                instances.append(self)
            def servo_config(self,pin,angle): pass
            def exit(self):
                self.exits+=1
                raise OSError('cleanup failed')
        camera=Camera()
        sdk=SimpleNamespace(ArducamCamera=lambda:camera,DepthData=Frame,
            Connection=SimpleNamespace(CSI=0),FrameType=SimpleNamespace(DEPTH=0),
            Control=SimpleNamespace(RANGE=0))
        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules',{'ArducamDepthCamera':sdk,
                'pyfirmata2':SimpleNamespace(Arduino=Board)}))
            stack.enter_context(patch.object(scan_track,'monotonic',side_effect=[0,2,2.1]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            for name in ('namedWindow','resizeWindow','imshow','destroyAllWindows'):
                stack.enter_context(patch.object(scan_track.cv2,name))
            with self.assertRaisesRegex(RuntimeError,'pin 9.*original write failed'):
                scan_track.run(parse_args(['--port','/dev/test']))
            instances[0].__del__()
        self.assertEqual(instances[0].exits,1)
        self.assertTrue(instances[0].sp.closed)
        self.assertTrue(camera.released and camera.stopped and camera.closed)


if __name__=='__main__':
    unittest.main()
