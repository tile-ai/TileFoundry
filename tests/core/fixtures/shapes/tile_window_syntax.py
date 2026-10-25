"""Authored tile-window syntax shared by parser and printer tests."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="scan_copy", target=CpuTarget())
class ScanCopy:
    @func
    def scan_copy(x: Tensor[(4, 4), "f32"]):
        out = tf.zeros(Tensor[(4, 4), "f32"])
        for row in tf.tile(4, 2):
            out = tf.insert_slice(out, x[row, :], (row, 0))
        return out


@module(entry="nested_scan_copy", target=CpuTarget())
class NestedScanCopy:
    @func
    def nested_scan_copy(x: Tensor[(4, 4), "f32"]):
        out = tf.zeros(Tensor[(4, 4), "f32"])
        for row in tf.tile(4, 2):
            for col in tf.tile(4, 2):
                out = tf.insert_slice(out, x[row, col], (row, col))
        return out


@module(entry="sequential_scan_copy", target=CpuTarget())
class SequentialScanCopy:
    @func
    def sequential_scan_copy(x: Tensor[(4, 4), "f32"]):
        out = tf.zeros(Tensor[(4, 4), "f32"])
        for row in tf.tile(4, 2):
            out = tf.insert_slice(out, x[row, :] + 1.0, (row, 0))
        for row in tf.tile(4, 2):
            out = tf.insert_slice(out, out[row, :] * 2.0, (row, 0))
        return out
