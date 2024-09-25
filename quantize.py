#!/usr/bin/env python

import json, sys, struct

with open(sys.argv[1]) as f:
    model = json.load(f)

class Emit:
    def __init__(self):
        self.buffer = bytearray()
        self.i16_format = struct.Struct("<h")
        self.i32_format = struct.Struct("<i")

    def put_i16(self, v, factor):
        v = round(v * factor)
        if v < -32767 or v > 32767:
            raise RuntimeError(f"{v} out of range of i16")
        self.buffer.extend(self.i16_format.pack(v))

    def put_many_i16(self, v, factor):
        if type(v) != list:
            return self.put_i16(v, factor)
        for x in v:
            self.put_many_i16(x, factor)

    def put_i32(self, v, factor):
        v = round(v * factor)
        if v < -2147483647 or v > 2147483647:
            raise RuntimeError(f"{v} out of range of i32")
        self.buffer.extend(self.i32_format.pack(v))

    def put_many_i32(self, v, factor):
        if type(v) != list:
            return self.put_i32(v, factor)
        for x in v:
            self.put_many_i32(x, factor)

def transpose(m):
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]

FT_UNIT = 256
L1_UNIT = 64

emit = Emit()

emit.put_many_i16(transpose(model["ft.weight"]), FT_UNIT)
emit.put_many_i16(model["ft.bias"], FT_UNIT)

emit.put_many_i16(model["l1.weight"], L1_UNIT)
emit.put_many_i32(model["l1.bias"], FT_UNIT * FT_UNIT * L1_UNIT)

with open(sys.argv[2], "wb") as f:
    f.write(emit.buffer)
