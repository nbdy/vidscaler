from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from threading import Event
from time import time
from typing import Callable, Optional

import cv2
import torch
from py_real_esrgan.model import RealESRGAN
from loguru import logger as log
from moviepy.editor import VideoFileClip
from numpy import array as np_array, ndarray
from tqdm import tqdm

SUPPORTED_VIDEO_FILE_TYPES = [".mp4", ".avi", ".flv", ".mkv"]
SUPPORTED_SCALES = [2, 4, 8]

ProgressCallback = Callable[[ndarray], None]


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', "--input", type=Path, default=Path("input"), help="Input directory or file")
    parser.add_argument('-o', "--output", type=Path, default=Path("output"), help="Output directory or file")
    parser.add_argument("-c", "--cpu", action="store_true", default=False, help="Use CPU mode")
    parser.add_argument("-s", "--scale", type=int, default=4, help="Upscaling factor")
    parser.add_argument("-m", "--model-directory", type=Path, default=Path.cwd() / "model")
    parser.add_argument("-g", "--gui", action="store_true", default=False, help="Run the GUI")
    parser.add_argument("-p", "--preview", action="store_true", default=False, help="Enable live preview")
    return parser.parse_args()


def scale_ok(scale: int) -> bool:
    ret = True
    if scale not in SUPPORTED_SCALES:
        log.error(f"Factor must be {' or'.join(str(s) for s in SUPPORTED_SCALES)}")
        ret = False
    return ret


class Upscaler:
    model: RealESRGAN = None
    progress_callback: Callable[[ndarray], None]
    fps: int = 0
    preview: bool = False

    def __init__(
            self,
            model_directory: Path,
            device: torch.device,
            scale: int = 4
    ) -> None:
        if not model_directory.exists():
            log.warning(f"Directory '{model_directory}' doesn't exist")
            log.debug(f"Creating directory '{model_directory}'")
            model_directory.mkdir(exist_ok=True)

        self.set_model(model_directory, device, scale)
        log.info(f"Initialized")

    def set_model(self, model_directory: Path, device: torch.device, scale: int = 4):
        model_path = model_directory / f"RealESRGAN_x{scale}.pth"
        log.debug(f"Loading RealESRGAN model '{model_path}'")
        self.model = RealESRGAN(device, scale)
        self.model.load_weights(model_path, download=True)

    def set_preview(self, value: bool) -> None:
        self.preview = value
        if not self.preview:
            cv2.destroyAllWindows()

    def upscale_video(
            self,
            input_path: Path,
            output_path: Path,
            stop_event: Optional[Event] = None,
            preview_event: Optional[Event] = None,
            frame_update_callback: Optional[Callable] = None,
            finish_callback: Optional[Callable] = None,
    ):
        input_video = VideoFileClip(str(input_path))
        total_frames = int(input_video.fps * input_video.duration)

        output_video_width = input_video.w * self.model.scale
        output_video_height = input_video.h * self.model.scale

        output_video = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            input_video.fps,
            (output_video_width, output_video_height)
        )

        log.info(f"Scaling {total_frames} ({input_video.w}x{input_video.h}) frames from '{input_path}' "
                 f"to '{output_path}' ({output_video_width}x{output_video_height})")

        pb = tqdm(total=total_frames, desc="Upscaling", unit="fps")
        start_time = time()
        for frame in input_video.iter_frames(fps=input_video.fps, dtype='uint8'):
            if stop_event and stop_event.is_set():
                break

            output = np_array(self.model.predict(frame))
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            predict_time = time() - start_time
            self.fps = pb.pos / predict_time
            output_video.write(output)

            if preview_event:
                if preview_event.is_set():
                    cv2.namedWindow("Preview", cv2.WINDOW_KEEPRATIO)
                    cv2.imshow("Preview", output)
                    cv2.resizeWindow("Preview", input_video.w, input_video.h)
                    cv2.waitKey(1)
                else:
                    cv2.destroyAllWindows()

            pb.update(1)

            if frame_update_callback:
                frame_update_callback(pb.n, total_frames)

        input_video.close()
        output_video.release()

        output_video = VideoFileClip(str(output_path))
        output_video.set_audio(input_video.audio)
        output_video.close()
        input_video.close()

        if finish_callback:
            finish_callback()

    def upscale_videos(
            self,
            input_videos: list[Path],
            output_path: Path,
            stop_event: Optional[Event] = None,
            preview_event: Optional[Event] = None,
            progress_callback: Callable[[Path], None] = None
    ) -> None:
        for input_video in input_videos:
            if progress_callback:
                progress_callback(input_video)
            self.upscale_video(
                input_video,
                output_path / input_video.name,
                stop_event,
                preview_event,
            )

    def upscale(
            self,
            input_path: Path,
            output_path: Path,
            stop_event: Optional[Event] = None,
            preview_event: Optional[Event] = None,
    ):
        if input_path.is_dir():
            for file in input_path.iterdir():
                if file.is_file() and file.suffix in SUPPORTED_VIDEO_FILE_TYPES:
                    self.upscale_video(
                        file,
                        output_path / file.name,
                        stop_event,
                        preview_event,
                    )
        else:
            self.upscale_video(
                input_path,
                output_path,
                stop_event,
                preview_event,
            )
