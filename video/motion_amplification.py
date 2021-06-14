import tensorflow as tf
import cv2

from common import plot, image as improc


def amplify_motion(video_path: str, save_gif: str):
    video_src = cv2.VideoCapture(video_path)

    # Load video
    video = []
    has_next_frame, frame = video_src.read()
    while has_next_frame:
        # Convert frame from BGR uint8 to RGB normalized float
        frame = improc.im2float(frame)
        video.append(frame)

        # Grab next frame
        has_next_frame, frame = video_src.read()

    plotter = plot.InteractivePlotter()
    # TODO Compute reference phase per level per pixel

    # TODO Compute delta phase per level per frame per pixel

    # TODO Bandpass Filter delta phases

    # TODO Amplify delta phases

    # TODO Apply delta to frequency domain levels

    # TODO Rebuild images from modified levels

    # Perform motion amplification
    for frame in video:
        plotter.update(frame)
    #plotter.finalize()
