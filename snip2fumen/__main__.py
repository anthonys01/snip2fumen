"""
process command from command line
"""
import argparse

from snip2fumen.shortcuts import listen_for_shortcuts
from snip2fumen.snipe2fumen import snipe_and_recog, recog_from_clipboard, recog_from_image


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
    parser.add_argument('-s', '--shortcuts',
                        dest='shortcuts',
                        action='store_true',
                        help='Listen for shortcuts '
                             '(<ctrl>+<shift>+D for clipboard2fumen,'
                             ' <ctrl>+Q to quit,'
                             ' <ctrl>+<shift>+C for experimental sniping)'
                        )

    args = parser.parse_args()

    to_clipboard = not args.not_to_clipboard
    if args.shortcuts:
        listen_for_shortcuts(to_clipboard)
    elif args.from_clipboard:
        recog_from_clipboard(to_clipboard)
    elif args.image:
        recog_from_image(args.image, to_clipboard)
    else:
        snipe_and_recog(to_clipboard)
