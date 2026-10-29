"""Bit-exact floating arithmetic for the AMX reference (round-to-nearest-even, DN).

AMX control semantics derived from corsix/amx, Copyright (c) 2022 Peter Cawley,
MIT licensed; see the accompanying reference source license.
"""

FORMATS = {16: (5, 10), 32: (8, 23), 64: (11, 52), "bf16": (8, 7)}


def _fmt(bits):
    e, f = FORMATS[bits]
    return e, f, (1 << (e - 1)) - 1, 1 << (e + f)


def fp_isnan(v, bits):
    e, f, _, _ = _fmt(bits)
    return ((v >> f) & ((1 << e) - 1)) == (1 << e) - 1 and bool(v & ((1 << f) - 1))


def default_nan(bits):
    e, f, _, _ = _fmt(bits)
    return (((1 << e) - 1) << f) | (1 << (f - 1))


def _unpack(v, bits):
    e, f, bias, sign = _fmt(bits)
    ex = (v >> f) & ((1 << e) - 1)
    m = v & ((1 << f) - 1)
    if ex == (1 << e) - 1:
        return (-1 if v & sign else 1), m, None
    return (-1 if v & sign else 1), m | ((1 << f) if ex else 0), (ex or 1) - bias - f


def _round_shift(n, s):
    if s <= 0:
        return n << -s
    q, r = divmod(n, 1 << s)
    h = 1 << (s - 1)
    return q + (r > h or (r == h and (q & 1)))


def _pack(sign, n, p, bits):
    e, f, bias, sm = _fmt(bits)
    sb = sm if sign < 0 else 0
    if not n:
        return sb
    exponent = n.bit_length() - 1 + p
    step = max(exponent - f, 1 - bias - f)
    q = _round_shift(n, step - p)
    if not q:
        return sb
    exponent = q.bit_length() - 1 + step
    if exponent > bias:
        return sb | (((1 << e) - 1) << f)
    if exponent < 1 - bias:
        return sb | q
    if q.bit_length() > f + 1:
        q >>= 1
    return sb | ((exponent + bias) << f) | (q & ((1 << f) - 1))


def convert_bits(v, src_bits, dst_bits):
    s, m, p = _unpack(v, src_bits)
    if p is None:
        if m:
            return default_nan(dst_bits)
        e, f, _, sm = _fmt(dst_bits)
        return (sm if s < 0 else 0) | (((1 << e) - 1) << f)
    return _pack(s, m, p, dst_bits)


def fp_fma(x, y, z, bits):
    sx, mx, px = _unpack(x, bits)
    sy, my, py = _unpack(y, bits)
    sz, mz, pz = _unpack(z, bits)
    e, f, _, sm = _fmt(bits)
    inf = ((1 << e) - 1) << f
    if any(fp_isnan(v, bits) for v in (x, y, z)):
        return default_nan(bits)
    if (px is None and py is not None and my == 0) or (py is None and px is not None and mx == 0):
        return default_nan(bits)
    if px is None or py is None:
        sp = sx * sy
        if pz is None and sp != sz:
            return default_nan(bits)
        return (sm if sp < 0 else 0) | inf
    if pz is None:
        return z
    sp = sx * sy
    mp = mx * my
    pp = px + py
    p = min(pp, pz)
    n = sp * (mp << (pp - p)) + sz * (mz << (pz - p))
    if n == 0:
        return sm if mp == mz == 0 and sp == sz == -1 else 0
    return _pack(-1 if n < 0 else 1, abs(n), p, bits)


def fp_mul(x, y, bits):
    sm = _fmt(bits)[3]
    return fp_fma(x, y, sm if bool(x & sm) ^ bool(y & sm) else 0, bits)


def fp_add(x, y, bits):
    _, f, bias, _ = _fmt(bits)
    return fp_fma(x, bias << f, y, bits)


def fp_compare(x, y, bits):
    if fp_isnan(x, bits) or fp_isnan(y, bits):
        return None
    sx, mx, px = _unpack(x, bits)
    sy, my, py = _unpack(y, bits)
    if not mx and px is not None and not my and py is not None:
        return 0
    if sx != sy:
        return -1 if sx < sy else 1
    sm = _fmt(bits)[3]
    ax = x & (sm - 1)
    ay = y & (sm - 1)
    return ((ax > ay) - (ax < ay)) * sx


def fp_minmax(x, z, bits, maximum=False):
    c = fp_compare(x, z, bits)
    if c is None:
        return default_nan(bits)
    if c == 0 and (x | z) & (_fmt(bits)[3] - 1) == 0:
        return (x & z) if maximum else (x | z)
    return x if (c > 0 if maximum else c < 0) else z


def bf16_to_f32(x):
    return (0x7FC0 if fp_isnan(x, "bf16") else x) << 16


def bf16_fma(x, y, z):
    def unpack(v):
        e = (v >> 7) & 255
        m = (v & 127) ^ 128
        if e == 0:
            m ^= 128
            e = m.bit_length() - 7 if m else -7
            m <<= 1 - e
        return e, m

    def sticky(v, s):
        return int(v != 0) if s >= 32 else (v >> s) | int(((v >> s) << s) != v)

    xe, xm = unpack(x)
    ye, ym = unpack(y)
    ze, zm = unpack(z)
    zs = z & 0x8000
    ps = (x ^ y) & 0x8000
    if xe == 255 or ye == 255 or ze == 255:
        if (
            (xe == 255 and (xm != 128 or ym == 0))
            or (ye == 255 and (ym != 128 or xm == 0))
            or (ze == 255 and zm != 128)
            or (ze == 255 and (xe == 255 or ye == 255) and zs != ps)
        ):
            return 0x7FC0
        return z if ze == 255 else ps | 0x7F80
    pm = xm * ym
    pe = xe + ye - 134
    if pm == 0:
        return z & (z if zm else ps)
    zm <<= 10
    ze -= 7
    pm <<= 3
    re = max(pe, ze)
    if pe < re:
        pm = sticky(pm, re - pe)
    if ze < re:
        zm = sticky(zm, re - ze)
    rs = ps if pm >= zm else zs
    if zs != rs:
        zm = (~zm) & 0xFFFFFFFF
    if ps != rs:
        pm = (~pm) & 0xFFFFFFFF
    rm = (zm + pm + (ps != zs)) & 0xFFFFFFFF
    if rm == 0:
        return zs & ps
    n = rm.bit_length() - 11
    re += n
    rm = rm << -n if n < 0 else sticky(rm, n)
    if re >= 255:
        return rs | 0x7F80
    if re <= 0:
        rm = sticky(rm, 1 - re)
        re = 0
    r = rs | (re << 7) | ((rm >> 3) & 127)
    return (r + (((rm & 7) + (r & 1)) > 4)) & 0xFFFF
