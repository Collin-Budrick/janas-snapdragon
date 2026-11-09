# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact Janas Q4_K/Q6_K/Q6_K_P activation-panel leaf kernels.

The versioned C ABI carries addresses as uint64_t words, including nullable
planes. Construct non-null Mojo pointers only after validating those words.
No allocation, runtime initialization, parallel executor, or model conversion
is involved. The C worker pool owns this call and all borrowed buffers.

Compile with --fp-mode contract=off. Explicit .fma() below matches the existing
C per-block chains; ordinary scale multiplication and final subtraction stay
separate. Weight blocks are decoded once for all token pairs and the odd tail.
"""

from std.memory import Pointer, unsafe_stack_allocation
from std.memory.unsafe import bitcast
from std.sys import llvm_intrinsic

comptime I4 = SIMD[.int32, 4]
comptime F4 = SIMD[.float32, 4]
comptime B16 = SIMD[.int8, 16]
comptime U16 = SIMD[.uint8, 16]
comptime Bytes = Pointer[UInt8, ImmUntrackedOrigin]
comptime SignedBytes = Pointer[Int8, ImmUntrackedOrigin]
comptime Ints = Pointer[Int32, MutUntrackedOrigin]
comptime Vectors = Pointer[B16, MutUntrackedOrigin]
comptime Q8_BYTES = 628


@inline(.always)
def _bytes(address: UInt64) -> Bytes:
    return Bytes(unsafe_from_address=Int(address))


@inline(.always)
def _load16(address: UInt64) -> U16:
    return _bytes(address).unsafe_load[width=16, alignment=1]()


@inline(.always)
def _h2f(address: UInt64) -> Float32:
    var p = Pointer[UInt16, ImmUntrackedOrigin](
        unsafe_from_address=Int(address)
    )
    return bitcast[.float16, 1](p.unsafe_load[alignment=1]()).cast[.float32]()


@inline(.always)
def _xf(address: UInt64) -> Float32:
    var p = Pointer[Float32, ImmUntrackedOrigin](
        unsafe_from_address=Int(address)
    )
    return p.unsafe_load[alignment=1]()


@inline(.always)
def _sum16(address: UInt64, index: Int) -> Int32:
    var p = Pointer[Int16, ImmUntrackedOrigin](
        unsafe_from_address=Int(address) + 260 + 2 * index
    )
    return p.unsafe_load[alignment=1]().cast[.int32]()


@inline(.always)
def _q4_correction(
    address: UInt64, mn0: Ints, mn1: Ints
) -> SIMD[.int32, 2]:
    # Widen each neighboring Int16 sum before addition, as C integer
    # promotion does. This also preserves arbitrary Int16 bsums metadata;
    # a non-widening ADDP could overflow for metadata outside Q8_K's usual
    # quantizer-produced bounds. No bsums2 representation is consulted.
    var p = Pointer[Int16, ImmUntrackedOrigin](
        unsafe_from_address=Int(address) + 260
    )
    var lo = llvm_intrinsic[
        "llvm.aarch64.neon.saddlp.v4i32.v8i16", I4
    ](p.unsafe_load[width=8, alignment=1]())
    var hi = llvm_intrinsic[
        "llvm.aarch64.neon.saddlp.v4i32.v8i16", I4
    ](p.unsafe_offset(8).unsafe_load[width=8, alignment=1]())
    var m00 = mn0.unsafe_load[width=4, alignment=1]()
    var m01 = mn0.unsafe_offset(4).unsafe_load[width=4, alignment=1]()
    var m10 = mn1.unsafe_load[width=4, alignment=1]()
    var m11 = mn1.unsafe_offset(4).unsafe_load[width=4, alignment=1]()
    # Only exact Int32 products/additions are regrouped. Each sum is below
    # 2**26 even for full-range Int16 metadata and six-bit Q4_K minima.
    return SIMD[.int32, 2](
        (lo * m00 + hi * m01).reduce_add(),
        (lo * m10 + hi * m11).reduce_add(),
    )


@inline(.always)
def _sdot(acc: I4, a: B16, b: B16) -> I4:
    return llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", I4](acc, a, b)


@inline(.always)
def _smmla(acc: I4, a: B16, b: B16) -> I4:
    return llvm_intrinsic["llvm.aarch64.neon.smmla.v4i32.v16i8", I4](acc, a, b)


@inline(.always)
def _pair8(a: B16, b: B16, high: Bool) -> B16:
    var aa = bitcast[.int64, 2](a)
    var bb = bitcast[.int64, 2](b)
    var lane = Int(high)
    return bitcast[.int8, 16](SIMD[.int64, 2](aa[lane], bb[lane]))


@inline(.always)
def _piece[LEVEL: Int, SHIFT: Int](a: U16, b: U16, c: U16) -> B16:
    var q = (a >> U16(SHIFT)) & 3
    comptime if LEVEL >= 2:
        q = (q << 2) | ((b >> U16(SHIFT)) & 3)
    comptime if LEVEL >= 3:
        q = (q << 2) | ((c >> U16(SHIFT)) & 3)
    return bitcast[.int8, 16](q)


@inline(.always)
def _unpack[
    TYPE: Int, LEVEL: Int
](
    row: UInt64,
    b: Int,
    p1: UInt64,
    p0: UInt64,
    q: Vectors,
    sc: Ints,
    mn: Ints,
) -> Tuple[Float32, Float32]:
    comptime if TYPE == 12:
        var w = row + UInt64(b * 144)
        comptime for j in range(4):
            var a = _load16(w + UInt64(16 + 32 * j))
            var z = _load16(w + UInt64(32 + 32 * j))
            q[unsafe_offset=4 * j] = bitcast[.int8, 16](a & 15)
            q[unsafe_offset=4 * j + 1] = bitcast[.int8, 16](z & 15)
            q[unsafe_offset=4 * j + 2] = bitcast[.int8, 16](a >> 4)
            q[unsafe_offset=4 * j + 3] = bitcast[.int8, 16](z >> 4)
        var s = _bytes(w + 4)
        comptime for j in range(4):
            sc[unsafe_offset=j] = Int32(s[unsafe_offset=j] & 63)
            mn[unsafe_offset=j] = Int32(s[unsafe_offset=j + 4] & 63)
        comptime for j in range(4, 8):
            sc[unsafe_offset=j] = Int32(
                (s[unsafe_offset=j + 4] & 15)
                | ((s[unsafe_offset=j - 4] >> 6) << 4)
            )
            mn[unsafe_offset=j] = Int32(
                (s[unsafe_offset=j + 4] >> 4) | ((s[unsafe_offset=j] >> 6) << 4)
            )
        return (_h2f(w), _h2f(w + 2))
    elif TYPE == 14:
        var w = row + UInt64(b * 210)
        comptime for n in range(2):
            comptime for h in range(2):
                var a = _load16(w + UInt64(64 * n + 16 * h))
                var z = _load16(w + UInt64(64 * n + 16 * h + 32))
                var hi = _load16(w + UInt64(128 + 32 * n + 16 * h))
                comptime g = 8 * n + h
                q[unsafe_offset=g] = (
                    bitcast[.int8, 16]((a & 15) | ((hi & 3) << 4)) - 32
                )
                q[unsafe_offset=g + 2] = (
                    bitcast[.int8, 16]((z & 15) | (((hi >> 2) & 3) << 4)) - 32
                )
                q[unsafe_offset=g + 4] = (
                    bitcast[.int8, 16]((a >> 4) | (((hi >> 4) & 3) << 4)) - 32
                )
                q[unsafe_offset=g + 6] = (
                    bitcast[.int8, 16]((z >> 4) | ((hi >> 6) << 4)) - 32
                )
        var scales = SignedBytes(unsafe_from_address=Int(w) + 192)
        comptime for g in range(16):
            sc[unsafe_offset=g] = Int32(scales[unsafe_offset=g])
        return (_h2f(w + 208), Float32(0))
    else:
        var w = row + UInt64(b * 82)
        comptime for n in range(2):
            comptime for h in range(2):
                comptime at = 32 * n + 16 * h
                var a = _load16(w + UInt64(at))
                var z = U16(0)
                var c = U16(0)
                comptime if LEVEL >= 2:
                    z = _load16(p1 + UInt64(b * 64 + at))
                comptime if LEVEL >= 3:
                    c = _load16(p0 + UInt64(b * 64 + at))
                comptime for part in range(4):
                    q[unsafe_offset=8 * n + h + 2 * part] = _piece[
                        LEVEL, 2 * part
                    ](a, z, c)
        var scales = SignedBytes(unsafe_from_address=Int(w) + 64)
        comptime for g in range(16):
            sc[unsafe_offset=g] = Int32(scales[unsafe_offset=g])
        return (_h2f(w + 80) * Float32(0.5), Float32(0))


@inline(.always)
def _rows[
    TYPE: Int, NV: Int, LEVEL: Int
](
    w: UInt64,
    row_bytes: Int,
    p1: UInt64,
    p0: UInt64,
    pb: Int,
    x: UInt64,
    xs: Int,
    qs: UInt64,
    nb: Int,
    rows: Int,
    y: UInt64,
    ys: Int,
):
    comptime PAIRS = NV // 2
    comptime GROUPS = 8 if TYPE == 12 else 16
    comptime LEN = 32 if TYPE == 12 else 16
    comptime FACTOR = 2 if LEVEL == 3 else 8 if LEVEL == 2 else 32
    comptime CENTER = 64 if LEVEL == 3 else 61 if LEVEL == 2 else 49
    var q0 = unsafe_stack_allocation[16, B16, alignment=16]()
    var q1 = unsafe_stack_allocation[16, B16, alignment=16]()
    var sc0 = unsafe_stack_allocation[16, Int32, alignment=16]()
    var sc1 = unsafe_stack_allocation[16, Int32, alignment=16]()
    var mn0 = unsafe_stack_allocation[8, Int32, alignment=16]()
    var mn1 = unsafe_stack_allocation[8, Int32, alignment=16]()
    var outp = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(y))
    for r in range(0, rows, 2):
        var acc = unsafe_stack_allocation[max(PAIRS, 1), F4, alignment=16]()
        var accm = unsafe_stack_allocation[max(PAIRS, 1), F4, alignment=16]()
        comptime for p in range(PAIRS):
            acc[unsafe_offset=p] = F4(0)
            accm[unsafe_offset=p] = F4(0)
        var odd0 = Float32(0)
        var odd1 = Float32(0)
        var oddm0 = Float32(0)
        var oddm1 = Float32(0)
        for b in range(nb):
            var d0 = _unpack[TYPE, LEVEL](
                w + UInt64(r * row_bytes),
                b,
                p1 + UInt64(r * pb),
                p0 + UInt64(r * pb),
                q0,
                sc0,
                mn0,
            )
            var d1 = _unpack[TYPE, LEVEL](
                w + UInt64((r + 1) * row_bytes),
                b,
                p1 + UInt64((r + 1) * pb),
                p0 + UInt64((r + 1) * pb),
                q1,
                sc1,
                mn1,
            )
            var sums = unsafe_stack_allocation[
                max(PAIRS, 1), I4, alignment=16
            ]()
            comptime for p in range(PAIRS):
                sums[unsafe_offset=p] = I4(0)
            var os0 = I4(0)
            var os1 = I4(0)
            var ox = x + UInt64(((NV - 1) * xs + b) * Q8_BYTES)
            comptime for g in range(GROUPS):
                var scales = I4(
                    sc0[unsafe_offset=g],
                    sc0[unsafe_offset=g],
                    sc1[unsafe_offset=g],
                    sc1[unsafe_offset=g],
                )
                var dots = unsafe_stack_allocation[
                    max(PAIRS, 1), I4, alignment=16
                ]()
                comptime for p in range(PAIRS):
                    dots[unsafe_offset=p] = I4(0)
                comptime for k in range(LEN // 8):
                    comptime at = g * LEN + k * 8
                    var a = _pair8(
                        q0[unsafe_offset=at // 16],
                        q1[unsafe_offset=at // 16],
                        Bool((at // 8) & 1),
                    )
                    comptime for p in range(PAIRS):
                        var bp = qs + UInt64(((p * nb + b) * 32 + at // 8) * 16)
                        dots[unsafe_offset=p] = _smmla(
                            dots[unsafe_offset=p],
                            a,
                            bitcast[.int8, 16](_load16(bp)),
                        )
                comptime for p in range(PAIRS):
                    sums[unsafe_offset=p] += dots[unsafe_offset=p] * scales
                comptime if NV & 1:
                    var od0 = I4(0)
                    var od1 = I4(0)
                    comptime for i in range(LEN // 16):
                        var aq = bitcast[.int8, 16](
                            _load16(ox + UInt64(4 + g * LEN + i * 16))
                        )
                        od0 = _sdot(
                            od0, q0[unsafe_offset=g * LEN // 16 + i], aq
                        )
                        od1 = _sdot(
                            od1, q1[unsafe_offset=g * LEN // 16 + i], aq
                        )
                    os0 += od0 * sc0[unsafe_offset=g]
                    os1 += od1 * sc1[unsafe_offset=g]
            var ds = F4(d0[0], d0[0], d1[0], d1[0])
            var ms = F4(d0[1], d0[1], d1[1], d1[1])
            comptime for p in range(PAIRS):
                var a = x + UInt64((2 * p * xs + b) * Q8_BYTES)
                var c = a + UInt64(xs * Q8_BYTES)
                var correction = I4(0)
                comptime if TYPE == 12:
                    var ca = _q4_correction(a, mn0, mn1)
                    var cc = _q4_correction(c, mn0, mn1)
                    correction = I4(ca[0], cc[0], ca[1], cc[1])
                elif TYPE == 214:
                    comptime for g in range(GROUPS):
                        var sa = _sum16(a, g)
                        var sb = _sum16(c, g)
                        var s0 = sc0[unsafe_offset=g]
                        var s1 = sc1[unsafe_offset=g]
                        correction += I4(s0 * sa, s0 * sb, s1 * sa, s1 * sb)
                var si = sums[unsafe_offset=p]
                comptime if TYPE == 214:
                    si = si * I4(FACTOR) - correction * I4(CENTER)
                var ad = _xf(a)
                var cd = _xf(c)
                var dx = F4(ad, cd, ad, cd)
                acc[unsafe_offset=p] = (ds * dx).fma(
                    si.cast[.float32](), acc[unsafe_offset=p]
                )
                comptime if TYPE == 12:
                    accm[unsafe_offset=p] = (ms * dx).fma(
                        correction.cast[.float32](), accm[unsafe_offset=p]
                    )
            comptime if NV & 1:
                var correction0 = Int32(0)
                var correction1 = Int32(0)
                comptime if TYPE == 12:
                    var co = _q4_correction(ox, mn0, mn1)
                    correction0 = co[0]
                    correction1 = co[1]
                elif TYPE == 214:
                    comptime for g in range(GROUPS):
                        var sx = _sum16(ox, g)
                        var s0 = sc0[unsafe_offset=g]
                        var s1 = sc1[unsafe_offset=g]
                        correction0 += sx * s0
                        correction1 += sx * s1
                var si0 = os0.reduce_add()
                var si1 = os1.reduce_add()
                comptime if TYPE == 214:
                    si0 = si0 * Int32(FACTOR) - correction0 * Int32(CENTER)
                    si1 = si1 * Int32(FACTOR) - correction1 * Int32(CENTER)
                var xd = _xf(ox)
                odd0 = (d0[0] * xd).fma(Float32(si0), odd0)
                odd1 = (d1[0] * xd).fma(Float32(si1), odd1)
                comptime if TYPE == 12:
                    oddm0 = (d0[1] * xd).fma(Float32(correction0), oddm0)
                    oddm1 = (d1[1] * xd).fma(Float32(correction1), oddm1)
        comptime for p in range(PAIRS):
            var v = acc[unsafe_offset=p]
            comptime if TYPE == 12:
                v -= accm[unsafe_offset=p]
            outp[unsafe_offset=r + 2 * p * ys] = v[0]
            outp[unsafe_offset=r + (2 * p + 1) * ys] = v[1]
            outp[unsafe_offset=r + 1 + 2 * p * ys] = v[2]
            outp[unsafe_offset=r + 1 + (2 * p + 1) * ys] = v[3]
        comptime if NV & 1:
            comptime if TYPE == 12:
                odd0 -= oddm0
                odd1 -= oddm1
            outp[unsafe_offset=r + (NV - 1) * ys] = odd0
            outp[unsafe_offset=r + 1 + (NV - 1) * ys] = odd1


@inline(.always)
def _width[
    TYPE: Int, LEVEL: Int
](
    w: UInt64,
    row_bytes: Int,
    p1: UInt64,
    p0: UInt64,
    pb: Int,
    x: UInt64,
    xs: Int,
    qs: UInt64,
    nb: Int,
    nv: Int,
    rows: Int,
    y: UInt64,
    ys: Int,
):
    comptime for n in range(1, 10):
        if nv == n:
            _rows[TYPE, n, LEVEL](
                w, row_bytes, p1, p0, pb, x, xs, qs, nb, rows, y, ys
            )
            return


@export("janas_mojo_abi_v1")
def janas_mojo_abi_v1() abi("C") -> UInt64:
    return UInt64(0x4A4D000100000001)


@export("janas_mojo_panel_rows_v1")
def janas_mojo_panel_rows_v1(
    type: Int32,
    level: Int32,
    w: UInt64,
    row_bytes: UInt64,
    p1: UInt64,
    p0: UInt64,
    pb: UInt64,
    x: UInt64,
    x_stride: UInt64,
    qs: UInt64,
    nb: UInt64,
    nv: UInt64,
    rows: UInt64,
    y: UInt64,
    ys: UInt64,
) abi("C") -> Int32:
    # Refusal must precede all stores, allowing the established C path to run.
    if type != 12 and type != 14 and type != 214:
        return 0
    if nv < 1 or nv > 9 or nb == 0 or rows == 0 or (rows & 1) != 0:
        return 0
    if w == 0 or x == 0 or y == 0 or (nv >= 2 and qs == 0):
        return 0
    if x_stride < nb or ys < rows:
        return 0
    var block_bytes = UInt64(144 if type == 12 else 210 if type == 14 else 82)
    if (
        nb > UInt64(0x7FFFFFFFFFFFFFFF) // block_bytes
        or row_bytes < nb * block_bytes
    ):
        return 0
    if nb > UInt64(0x7FFFFFFFFFFFFFFF) // 64 or pb < nb * 64:
        return 0
    if row_bytes > UInt64(0x7FFFFFFFFFFFFFFF) // rows:
        return 0
    if x_stride > UInt64(0x7FFFFFFFFFFFFFFF) // (nv * UInt64(Q8_BYTES)):
        return 0
    if ys > UInt64(0x7FFFFFFFFFFFFFFF) // (nv * 4):
        return 0
    if type != 214:
        if level != 0 or p1 != 0 or p0 != 0:
            return 0
    else:
        if level < 1 or level > 3:
            return 0
        if (
            (level == 1 and (p1 != 0 or p0 != 0))
            or (level == 2 and (p1 == 0 or p0 != 0))
            or (level == 3 and (p1 == 0 or p0 == 0))
        ):
            return 0
        if pb > UInt64(0x7FFFFFFFFFFFFFFF) // rows:
            return 0
    if type == 12:
        _width[12, 0](
            w,
            Int(row_bytes),
            p1,
            p0,
            Int(pb),
            x,
            Int(x_stride),
            qs,
            Int(nb),
            Int(nv),
            Int(rows),
            y,
            Int(ys),
        )
    elif type == 14:
        _width[14, 0](
            w,
            Int(row_bytes),
            p1,
            p0,
            Int(pb),
            x,
            Int(x_stride),
            qs,
            Int(nb),
            Int(nv),
            Int(rows),
            y,
            Int(ys),
        )
    elif level == 1:
        _width[214, 1](
            w,
            Int(row_bytes),
            p1,
            p0,
            Int(pb),
            x,
            Int(x_stride),
            qs,
            Int(nb),
            Int(nv),
            Int(rows),
            y,
            Int(ys),
        )
    elif level == 2:
        _width[214, 2](
            w,
            Int(row_bytes),
            p1,
            p0,
            Int(pb),
            x,
            Int(x_stride),
            qs,
            Int(nb),
            Int(nv),
            Int(rows),
            y,
            Int(ys),
        )
    else:
        _width[214, 3](
            w,
            Int(row_bytes),
            p1,
            p0,
            Int(pb),
            x,
            Int(x_stride),
            qs,
            Int(nb),
            Int(nv),
            Int(rows),
            y,
            Int(ys),
        )
    return 1
