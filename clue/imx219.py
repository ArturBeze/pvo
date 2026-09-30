#!/usr/bin/python3

# Normally the QtGlPreview implementation is recommended as it benefits
# from GPU hardware acceleration.

import cv2
import sys, os
import time, libcamera
from pprint import pprint

from picamera2 import Picamera2, Preview, Metadata
from picamera2.encoders import H264Encoder


def time_elapsed(start_time, event):
    time_now = time.time()
    duration = (time_now - start_time) * 1000
    duration=round(duration, 2)
    print (">>> ", duration, " ms (" ,event, ")")


def main():

    encoder = H264Encoder(10000000)
    output = "test.h264"

    picam2 = Picamera2()
    picam2.start_preview(Preview.QTGL)
    # picam2.start_recording(encoder, output)


    # camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    # picam2.configure(camera_config)


    # video_config = picam2.create_video_configuration(main={"size": (1640, 1232)}, controls={"FrameDurationLimits": (40000, 40000)}, lores={"size": (640, 480)})
    # video_config = picam2.create_video_configuration(main={"size": (640, 480), "format": 'RGB888'}, raw=picam2.sensor_modes[0], buffer_count=8)
    # picam2.configure(video_config)


    preview_config = picam2.create_preview_configuration()
    # preview_config = picam2.create_preview_configuration(main={"size": (1600, 1200), "format": "XRGB8888"}, lores={"size": (640, 480), "format": "YUV420"}, display="lores")
    # preview_config["transform"] = libcamera.Transform(hflip=1, vflip=1)
    picam2.configure(preview_config)


    pprint(picam2.sensor_modes)
    print('#' * 30)

    picam2.set_controls({"FrameRate": 206})

    picam2.start()
    time.sleep(5)
    picam2.title_fields = ["ExposureTime", "AnalogueGain"]

    # Получаем metadata текущего кадра
    # picam2.capture_metadata()["FrameDuration"]
    # picam2.capture_metadata()["SensorTimestamp"]
    metadata = Metadata(picam2.capture_metadata())
    pprint(picam2.capture_metadata())
    print('#' * 30)

    captureNanoSEC = str(metadata.SensorTimestamp)
    captureUSEC0 = int(captureNanoSEC[0:(len(captureNanoSEC) - 3)])
    captureUSEC = captureUSEC0

    input("Press any key: ")

    while True:
        t1 = time.time()
        request = picam2.capture_request()
        # im = picam2.capture_array()
        frame = request.make_array('main')
        metadata = request.get_metadata()
        request.release()
        time_elapsed(t1, "Capture image (t1)")

        captureUSECLast = captureUSEC
        captureNanoSEC = str(metadata["SensorTimestamp"])
        captureUSEC = int(captureNanoSEC[0:(len(captureNanoSEC) - 3)])

        print(f"TS = {metadata['SensorTimestamp']} fps = {round(1000000 / (captureUSEC - captureUSECLast), 2)} ({round((captureUSEC - captureUSECLast) / 1000000, 5)} ms)")

        t2 = time.time()
        cv2_im = frame
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        time_elapsed(t2, "Image processing (t2)")

        cv2.imshow('frame', cv2_im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # picam2.stop_recording()
    picam2.stop_preview()

    picam2.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)