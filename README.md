# video-upscaler

[![](http://github-actions.40ants.com/nbdy/vidscaler/matrix.svg)](https://github.com/nbdy/vidscaler)

upscale (x2/x4/x8) a video or a directory of videos using the command line

## dependencies
- moviepy
- loguru
- opencv
- torch
- numpy
- tqdm
- [Real-ESRGAN](https://github.com/sberbank-ai/Real-ESRGAN)


## how to ..

### .. install

```shell
# from pypi
pip install vidscaler

# from git repo
pip install git+https://github.com/nbdy/vidscaler

# or download a release from https://github.com/nbdy/vidscaler/releases
```

### .. run

#### .. gui

```shell
vidscaler-gui
# or
vidscaler --gui
```

#### cli

```shell
vidscaler --help
usage: vidscaler [-h] -i INPUT -o OUTPUT [-c] [-s SCALE]
                   [-m MODEL_DIRECTORY] [-g] [-p]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input directory or file
  -o OUTPUT, --output OUTPUT
                        Output directory or file
  -c, --cpu             Use CPU mode
  -s SCALE, --scale SCALE
                        Upscaling factor
  -m MODEL_DIRECTORY, --model-directory MODEL_DIRECTORY
  -g, --gui             Run the GUI
  -p, --preview         Enable live preview
```
