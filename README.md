# snip2fumen

Python tool to screenshot Tetris boards and generate fumen strings from it

![yPKOXhMqX4](https://user-images.githubusercontent.com/35337929/153570060-19287369-a00c-45a6-8fdf-66b571701d0b.gif)


## Description

snip2fumen is a python tool to take screenshots of your Tetris board and generate a fumen code with it (from [tetris-fumen](https://github.com/knewjade/tetris-fumen))

It was tested with [jstris](https://jstris.jezevec10.com), four-tris and [tetr.io](https://tetr.io) with default skins.
It should work for tetris games using the standard tetris color schemes + having a visible grid (which means this tool is not compatible with TGM games).

It should work even if you don't take a nicely cropped screenshot, or even if you don't take a full board.

## Installation

Developed and tested on python 3.9 (it should work with python 3.6+).

The dependencies are listed in the requirements.txt files :
* opencv-python
* numpy
* pyqt5
* pyperclip

Use pip to install them :
```
pip install -r requirements.txt
```

## Usage

Once installation is done, run the snip2fumen module (you need to be in the same folder as this README)

```
python -m snip2fumen
```

Then use your mouse to select your game board (like any sniping tool).
The resulting fumen link will be printed as a result, and will be set in you clipboard.

Should be working on all operating system, but was only tested on Windows 10.

Additionally, if you can run it, there is a standalone snip2fumen.exe executable in dist/.
