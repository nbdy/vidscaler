import threading
from pathlib import Path
from tkinter import filedialog, StringVar
from typing import Dict, List

import torch
from customtkinter import CTk, CTkLabel, CTkEntry, CTkButton, CTkComboBox, CTkProgressBar, CTkCheckBox, BooleanVar

from vidscaler import Upscaler, log, SUPPORTED_VIDEO_FILE_TYPES, parse_args

LABEL_WIDTH = 32

SCALER_LUT = {"x2": 2, "x4": 4, "x8": 8}


class GUI(CTk):
    entries: Dict[str, CTkEntry]
    combos: Dict[str, CTkComboBox]

    running: bool = False
    input_files: List[Path] = []

    upscaler: Upscaler = None
    processing_thread: threading.Thread = None
    stop_event: threading.Event = threading.Event()
    preview_event: threading.Event = threading.Event()
    current_file_index: int = 1
    current_scale: int = 4

    def __init__(self, args):
        CTk.__init__(self)
        self.title("VidScaler")
        self.resizable(False, False)

        self.paths = {
            "input": args.input,
            "output": args.output,
            "model": args.model_directory
        }
        self.entries = {
            "input": CTkEntry(self),
            "output": CTkEntry(self),
            "model": CTkEntry(self)
        }
        self.buttons = {
            "input": CTkButton(self, 20, text="Select", command=lambda: self.show_file_selector("input")),
            "output": CTkButton(self, 20, text="Select", command=lambda: self.show_file_selector("output")),
            "model": CTkButton(self, 20, text="Select", command=lambda: self.show_file_selector("model")),
            "action": CTkButton(self, 20, text="Start", command=self.action_button_clicked),
        }
        self.combos = {
            "device": CTkComboBox(self, 96, values=["cpu"] if args.cpu else ["cpu", "cuda"],
                                  variable=StringVar(value="cpu" if args.cpu else "cuda"),
                                  command=lambda x: self.set_combo("device", x)),
            "scale": CTkComboBox(self, 96, values=list(SCALER_LUT.keys()),
                                 variable=StringVar(value=f"x{args.scale}"),
                                 command=lambda x: self.set_combo("scale", x))
        }
        self.progressbars = {
            "file": CTkProgressBar(self, orientation="horizontal", mode="determinate"),
            "total": CTkProgressBar(self, orientation="horizontal", mode="determinate")
        }
        self.progressbars["file"].set(0)
        self.progressbars["total"].set(0)
        self.progressbar_labels = {
            "file": CTkLabel(self, LABEL_WIDTH, text="File:\t0 / 0"),
            "total": CTkLabel(self, LABEL_WIDTH, text="Total:\t0 / 0"),
            "fps": CTkLabel(self, LABEL_WIDTH, text="FPS:\t0"),
            "file_percent": CTkLabel(self, LABEL_WIDTH, text="0 %"),
            "total_percent": CTkLabel(self, LABEL_WIDTH, text="0 %"),
        }
        self.checkboxes = {
            "preview": CTkCheckBox(self, text="Preview", variable=BooleanVar(value=args.preview),
                                   onvalue=True, offvalue=False,
                                   command=lambda: self.set_checkbox("preview")),
        }

        self.set_entry("input", args.input)
        self.set_entry("output", args.output)
        self.set_entry("model", args.model_directory)

        self.set_upscaler()

        log.info("Initialized")
        self.show_main_screen()

    def set_upscaler(self):
        model_path = self.paths["model"]
        device = self.combos["device"].get()
        scale = int(self.combos["scale"].get().strip("x"))

        self.upscaler = Upscaler(model_path, torch.device(device), scale)

    def set_combo(self, key: str, value: str):
        log.info(f"Set {key} to {value}")
        if key in ["device", "scale"]:
            self.set_upscaler()

    def set_checkbox(self, key: str):
        value = self.checkboxes[key].get()

        if key == "preview":
            if value:
                self.preview_event.set()
            else:
                self.preview_event.clear()

    def _update_file_count(self, key: str):
        self.input_files = []

        path = self.paths[key]

        if path.exists():
            if path.is_file() and path.suffix in SUPPORTED_VIDEO_FILE_TYPES:
                self.input_files.append(self.paths[key])
            else:
                for fp in path.iterdir():
                    if fp.suffix in SUPPORTED_VIDEO_FILE_TYPES:
                        self.input_files.append(fp)
        else:
            log.error(f"Path '{path}' does not exist")

        input_file_count = len(self.input_files)
        log.debug(f"{input_file_count} files in {self.paths[key]}")

        self.progressbars["file"].value = 0
        self.progressbar_labels["file"].configure(text=f"Frame:\t0 / 0")
        self.progressbar_labels["file_percent"].configure(text=f"0 %")

        self.progressbars["total"].value = 0
        self.progressbar_labels["total"].configure(text=f"File:\t0 / {input_file_count}")
        self.progressbar_labels["total_percent"].configure(text=f"0 %")

    def set_entry(self, key: str, value: Path):
        self.paths[key] = value
        log.info(f"Selected {key} directory: {value}")

        self.entries[key].configure(textvariable=StringVar(value=str(value)))

        if key in ["input"]:
            self._update_file_count(key)

    def show_file_selector(self, key: str):
        self.set_entry(key, Path(filedialog.askdirectory()))

    def set_frame_progress(self, frame: int, max_frames: int):
        progress = frame / max_frames
        self.progressbars["file"].set(progress)
        self.progressbar_labels["file"].configure(text=f"Frame:\t{frame} / {max_frames}")
        self.progressbar_labels["file_percent"].configure(text=f"{progress * 100:.2f} %")

    def update_file_progress(self):
        progress = self.current_file_index / len(self.input_files)
        self.progressbars["total"].set(progress)
        self.progressbar_labels["total"].configure(text=f"File:\t{self.current_file_index} / {len(self.input_files)}")
        self.progressbar_labels["total_percent"].configure(text=f"{progress * 100:.2f} %")

    def start_stop_upscaling(self):
        def frame_update_callback(frame: int, max_frames: int):
            self.set_frame_progress(frame, max_frames)

        def file_finished_callback():
            self.current_file_index += 1
            self.update_file_progress()

            if self.current_file_index < len(self.input_files):
                self.start_stop_upscaling()

        self._update_file_count("input")

        if self.running:
            self.stop_event.clear()
            log.info("Starting upscaling")
            video = self.input_files[self.current_file_index - 1]
            self.update_file_progress()
            self.processing_thread = threading.Thread(
                target=self.upscaler.upscale_video,
                args=(
                    video,
                    self.paths["output"] / video.name,
                    self.stop_event,
                    self.preview_event,
                    frame_update_callback,
                    file_finished_callback,
                )
            )
            self.processing_thread.start()
        else:
            log.info("Stopping upscaling")
            self.stop_event.set()
            self.stop_event.wait()
            self.current_file_index = 1
            self.progressbars["file"].set(0)
            self.progressbars["total"].set(0)
            self.update_file_progress()
            self._update_file_count("input")
            log.info("Stopped upscaling")

    def action_button_clicked(self):
        self.running = not self.running

        self.buttons["action"].configure(text=("Stop" if self.running else "Start"))

        entry_state = "readonly" if self.running else "normal"
        self.entries["input"].configure(state=entry_state)

        button_state = "disabled" if self.running else "normal"
        self.buttons["input"].configure(state=button_state)
        self.buttons["output"].configure(state=button_state)
        self.buttons["model"].configure(state=button_state)
        self.combos["device"].configure(state=button_state)
        self.combos["scale"].configure(state=button_state)

        self.start_stop_upscaling()

    def show_main_screen(self):
        log.debug("Main screen")
        # row 1
        input_directory_label = CTkLabel(self, LABEL_WIDTH, text="Input Directory:")
        input_directory_label.grid(row=0, column=0, columnspan=1, padx=4, pady=4, sticky="w")
        self.entries["input"].grid(row=0, column=1, columnspan=4, padx=4, pady=4, sticky="we")
        self.buttons["input"].grid(row=0, column=5, columnspan=1, padx=4, pady=4, sticky="we")

        # row 2
        output_directory_label = CTkLabel(self, LABEL_WIDTH, text="Output Directory:")
        output_directory_label.grid(row=1, column=0, columnspan=1, padx=4, pady=4, sticky="w")
        self.entries["output"].grid(row=1, column=1, columnspan=4, padx=4, pady=4, sticky="we")
        self.buttons["output"].grid(row=1, column=5, columnspan=1, padx=4, pady=4, sticky="we")

        # row 3
        model_directory_label = CTkLabel(self, LABEL_WIDTH, text="Model Directory:")
        model_directory_label.grid(row=2, column=0, columnspan=1, padx=4, pady=4, sticky="w")
        self.entries["model"].grid(row=2, column=1, columnspan=4, padx=4, pady=4, sticky="we")
        self.buttons["model"].grid(row=2, column=5, columnspan=1, padx=4, pady=4, sticky="we")

        # row 4
        self.progressbar_labels["file"].grid(row=3, column=0, columnspan=1, padx=4, pady=4, sticky="w")
        self.progressbars["file"].grid(row=3, column=1, columnspan=4, padx=4, pady=4, sticky="we")
        self.progressbar_labels["file_percent"].grid(row=3, column=5, columnspan=1, padx=4, pady=4, sticky="we")

        # row 5
        self.progressbar_labels["total"].grid(row=4, column=0, columnspan=1, padx=4, pady=4, sticky="w")
        self.progressbars["total"].grid(row=4, column=1, columnspan=4, padx=4, pady=4, sticky="we")
        self.progressbar_labels["total_percent"].grid(row=4, column=5, padx=4, pady=4, sticky="we")

        # row 6
        self.buttons["action"].grid(row=5, column=0, columnspan=1, padx=4, pady=4, sticky="we")

        device_label = CTkLabel(self, text="Device")
        device_label.grid(row=5, column=1, columnspan=1, padx=4, pady=4, sticky="w")
        self.combos["device"].grid(row=5, column=2, columnspan=1, padx=4, pady=4, sticky="w")

        scale_label = CTkLabel(self, text="Scale")
        scale_label.grid(row=5, column=3, columnspan=1, padx=4, pady=4, sticky="w")
        self.combos["scale"].grid(row=5, column=4, columnspan=1, padx=4, pady=4, sticky="w")

        self.checkboxes["preview"].grid(row=5, column=5, columnspan=1, padx=4, pady=4, sticky="we")

        self.entries["output"].configure(state="disabled")
        self.entries["model"].configure(state="disabled")


if __name__ == '__main__':
    args = parse_args()
    GUI(args).mainloop()
