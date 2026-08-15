"""Install chosen attract demos into N++'s local attract cache so the game
plays exactly those, in a loop.

    python install_attracts.py 10131255 10138317 10138288 10131738
    python install_attracts.py --restore

The game reads Documents/Metanet/N++/attract/<n>, where <n> is an arbitrary
integer name; file bytes are identical to the corpus files in ATTRACT_SRC.
The game cycles the folder in name order, so with N files installed you get a
clean N-demo loop and Simulator::reset marks each boundary.

N++ must be restarted after changing the folder -- it enumerates once at start.
"""
import os
import shutil
import sys

ATTRACT_SRC = r"D:\attracts"
CACHE = os.path.join(os.path.expanduser("~"), "Documents", "Metanet", "N++",
                     "attract")
BACKUP = CACHE + ".bak-preRE"


def restore():
    if not os.path.isdir(BACKUP):
        print("no backup at %s" % BACKUP)
        return
    if os.path.isdir(CACHE):
        shutil.rmtree(CACHE)
    os.rename(BACKUP, CACHE)
    print("restored %d files" % len(os.listdir(CACHE)))


def install(ids):
    if not os.path.isdir(BACKUP):
        os.rename(CACHE, BACKUP)
        print("backed up original cache -> %s" % BACKUP)
    if os.path.isdir(CACHE):
        shutil.rmtree(CACHE)
    os.makedirs(CACHE)
    for i, demo in enumerate(ids):
        src = os.path.join(ATTRACT_SRC, demo)
        if not os.path.exists(src):
            print("MISSING %s" % src)
            continue
        shutil.copyfile(src, os.path.join(CACHE, str(i)))
        print("  slot %d <- %s" % (i, demo))
    print("installed %d demos; restart N++" % len(os.listdir(CACHE)))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--restore":
        restore()
    else:
        install(args)
