"""
download the tiny-shakespeare dataset to data/input.txt

    python data/prepare.py
"""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = os.path.join(os.path.dirname(__file__), "input.txt")


def main():
    if os.path.exists(OUT):
        print(f"already have {OUT}")
        return
    print(f"downloading -> {OUT}")
    urllib.request.urlretrieve(URL, OUT)
    print(f"done: {os.path.getsize(OUT)} bytes")


if __name__ == "__main__":
    main()
