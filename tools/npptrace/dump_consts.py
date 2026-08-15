#!/usr/bin/env python3
"""Read arbitrary constants out of npp.dll by virtual address.

    python tools/npptrace/dump_consts.py 0x10571cd8 0x10586ca0 ...
    python tools/npptrace/dump_consts.py --dll /path/to/npp.dll 0x10571cd8
"""
from __future__ import annotations

import struct
import sys

from dump_ragdoll_tables import PE, DEFAULT_DLL


def main():
    args = list(sys.argv[1:])
    dll = DEFAULT_DLL
    if args and args[0] == "--dll":
        dll = args[1]
        args = args[2:]
    pe = PE(dll)
    for a in args:
        va = int(a, 0)
        raw = pe.read(va, 8)
        d, = struct.unpack("<d", raw)
        i0, i1 = struct.unpack("<2i", raw)
        f0, f1 = struct.unpack("<2f", raw)
        print("0x%08X  %s  d=%-24r i=(%d, %d) f=(%r, %r)"
              % (va, raw.hex(" "), d, i0, i1, f0, f1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
