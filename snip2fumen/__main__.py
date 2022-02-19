"""
process command from command line
"""
import argparse
import cv2
from PIL import ImageGrab
import numpy as np

from snip2fumen.snipe2fumen import snipe_and_recog
from snip2fumen import recog


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-fc", "--from-clipboard",
                        dest="from_clipboard",
                        action='store_true',
                        help="Directly take the image to manipulate from the clipboard "
                             "(takes priority over given image path)"
                        )
    parser.add_argument("-nc", "--not-to-clipboard",
                        dest="not_to_clipboard",
                        action='store_true',
                        help="Do not set result into clipboard"
                        )
    parser.add_argument('-f', '--file',
                        dest='image',
                        help='Directly use given image',
                        type=str
                        )

    args = parser.parse_args()

    to_clipboard = not args.not_to_clipboard
    if args.from_clipboard:
        im = ImageGrab.grabclipboard()
        if im:
            im_array = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            recog.recog_image(im_array, to_clipboard)
        else:
            print("No image was found in clipboard")
    elif args.image:
        im = cv2.imread(args.image)
        if im is not None:
            recog.recog_image(im, to_clipboard)
        else:
            print(f"Could not find any image at path {args.image}")
    else:
        snipe_and_recog(to_clipboard)
