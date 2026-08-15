#!/usr/bin/env python3
"""Dump the ragdoll's static tables straight out of npp.dll.

    python tools/npptrace/dump_ragdoll_tables.py [path/to/npp.dll]

No running game needed: every table the ragdoll reads is initialised data in
the module's .data section, so the file bytes are the ground truth.

Addresses come from the decompiled ragdoll (docs/decomp):

    0x10586730  stick table   {i32 a; i32 b; f64 rest_len}  x 11, per ragdoll kind
    0x10586890  angle table   {i32 c; i32 a; i32 b; i32 _; f64 target; f64 tol} x 9
    0x10586AD0  i32 bone count   per kind
    0x10586AD8  i32 angle count  per kind
    0x10586AE0  f64 drag
    0x10586518  f64 gravity
"""
from __future__ import annotations

import struct
import sys

DEFAULT_DLL = "/mnt/d/Games/Steam/steamapps/common/N++/npp.dll"

STICKS = 0x10586730
ANGLES = 0x10586890
BONE_COUNTS = 0x10586AD0
ANGLE_COUNTS = 0x10586AD8
DRAG = 0x10586AE0
GRAVITY = 0x10586518

NUM_KINDS = 2
NUM_STICKS = 11
NUM_ANGLES = 9
STICK_STRIDE = 16
ANGLE_STRIDE = 32


class PE:
    def __init__(self, path):
        self.buf = open(path, "rb").read()
        e_lfanew = struct.unpack_from("<I", self.buf, 0x3C)[0]
        assert self.buf[e_lfanew:e_lfanew + 4] == b"PE\0\0", "not a PE"
        coff = e_lfanew + 4
        n_sections, = struct.unpack_from("<H", self.buf, coff + 2)
        opt_size, = struct.unpack_from("<H", self.buf, coff + 16)
        opt = coff + 20
        magic, = struct.unpack_from("<H", self.buf, opt)
        assert magic == 0x10B, "expected PE32, got 0x%X" % magic
        self.image_base, = struct.unpack_from("<I", self.buf, opt + 28)
        self.sections = []
        s = opt + opt_size
        for i in range(n_sections):
            o = s + i * 40
            name = self.buf[o:o + 8].rstrip(b"\0").decode("latin1")
            vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", self.buf, o + 8)
            self.sections.append((name, vaddr, vsize, rawptr, rawsize))

    def off(self, va):
        rva = va - self.image_base
        for name, vaddr, vsize, rawptr, rawsize in self.sections:
            if vaddr <= rva < vaddr + max(vsize, rawsize):
                delta = rva - vaddr
                if delta >= rawsize:
                    raise ValueError("VA 0x%X is in %s but past raw data (bss)"
                                     % (va, name))
                return rawptr + delta
        raise ValueError("VA 0x%X not in any section" % va)

    def read(self, va, n):
        o = self.off(va)
        return self.buf[o:o + n]

    def i32(self, va):
        return struct.unpack("<i", self.read(va, 4))[0]

    def f64(self, va):
        return struct.unpack("<d", self.read(va, 8))[0]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DLL
    pe = PE(path)
    print("%s  image_base=0x%08X" % (path, pe.image_base))
    for name, vaddr, vsize, rawptr, rawsize in pe.sections:
        print("   %-8s rva=0x%08X vsize=0x%-8X raw=0x%-8X rawsize=0x%X"
              % (name, vaddr, vsize, rawptr, rawsize))
    print()

    print("gravity = %r" % pe.f64(GRAVITY))
    print("drag    = %r" % pe.f64(DRAG))
    print()

    for kind in range(NUM_KINDS):
        nb = pe.i32(BONE_COUNTS + 4 * kind)
        na = pe.i32(ANGLE_COUNTS + 4 * kind)
        print("=== ragdoll kind %d: %d bones, %d angle constraints ===" % (kind, nb, na))
        print("  sticks (a, b, rest_len):")
        segs = []
        for i in range(NUM_STICKS):
            va = STICKS + kind * NUM_STICKS * STICK_STRIDE + i * STICK_STRIDE
            a = pe.i32(va)
            b = pe.i32(va + 4)
            ln = pe.f64(va + 8)
            segs.append((a, b, ln))
            print("    [%2d] %2d -> %2d   len=%r" % (i, a, b, ln))
        print("  python: SEGS = %r" % (tuple((a, b) for a, b, _ in segs),))
        print("  python: SEG_LENGTHS = %r" % (tuple(ln for _, _, ln in segs),))
        print("  angles (centre, arm_a, arm_b, spare, target_rad, tol_rad):")
        for i in range(NUM_ANGLES):
            va = ANGLES + kind * NUM_ANGLES * ANGLE_STRIDE + i * ANGLE_STRIDE
            c, a, b, spare = struct.unpack("<4i", pe.read(va, 16))
            target, tol = struct.unpack("<2d", pe.read(va + 16, 16))
            mark = "" if i < na else "   (unused)"
            print("    [%2d] c=%2d a=%2d b=%2d spare=%2d target=%-22r tol=%r%s"
                  % (i, c, a, b, spare, target, tol, mark))
        print()

    print("raw stick region 0x%08X..0x%08X:" % (STICKS, ANGLES))
    raw = pe.read(STICKS, ANGLES - STICKS)
    for i in range(0, len(raw), 16):
        print("  0x%08X  %s" % (STICKS + i, raw[i:i + 16].hex(" ")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
