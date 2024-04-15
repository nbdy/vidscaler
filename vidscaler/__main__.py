from pathlib import Path

import torch

from vidscaler import log, Upscaler, parse_args


def gui(args):
    from vidscaler.gui import GUI
    GUI(args).mainloop()


def cli(args):
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        log.error(f"Input path '{input_path}' does not exist")
        return

    if not output_path.exists():
        log.error(f"Output path '{output_path}' does not exist")
        return

    if input_path.is_dir() and output_path.is_file():
        log.error("Input is a directory but output is a file!")
        return

    if input_path.is_file() and output_path.is_dir():
        log.error("Output is a directory but input is a file!")
        return

    device = torch.device("cpu" if args.cpu else "cuda")
    upscaler = Upscaler(args.model_directory, device, args.scale)
    upscaler.upscale(input_path, output_path)


def main():
    args = parse_args()

    if args.gui:
        gui(args)
    else:
        cli(args)

    log.info("Exiting")


if __name__ == '__main__':
    main()
