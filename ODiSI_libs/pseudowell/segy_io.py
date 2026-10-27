"""SEG-Y read/write helpers (segyio)."""
from pathlib import Path

import numpy as np
import segyio


def resolve_path(path, data_dir="."):
    """Return an existing path, matching the file name case-insensitively.

    Some input file names differ only in case between systems.
    """
    p = Path(data_dir) / path
    if p.exists():
        return p
    matches = [q for q in p.parent.iterdir() if q.name.lower() == p.name.lower()]
    if matches:
        return matches[0]
    raise FileNotFoundError(p)


def read_segy(path):
    """Read a 2D line. Returns (data[nt, nx], dt_seconds, segyio handle info)."""
    with segyio.open(str(path), ignore_geometry=True) as f:
        data = segyio.tools.collect(f.trace[:]).T.astype(np.float64)
        dt = segyio.tools.dt(f) / 1e6
    return data, dt


def write_segy(path, data, dt, template=None):
    """Write data[ns, ntr] as IEEE-float SEG-Y with sample interval dt (s).

    If ``template`` (an existing SEG-Y with the same trace count) is given,
    its textual header and trace headers (CDP, coordinates, ...) are copied.
    """
    ns, ntr = data.shape
    spec = segyio.spec()
    spec.samples = np.arange(ns) * dt * 1000.0
    spec.format = 5
    spec.tracecount = ntr
    with segyio.create(str(path), spec) as dst:
        src = segyio.open(str(template), ignore_geometry=True) if template else None
        try:
            if src is not None:
                dst.text[0] = src.text[0]
            for i in range(ntr):
                if src is not None and i < src.tracecount:
                    dst.header[i] = src.header[i]
                dst.header[i].update({
                    segyio.TraceField.TRACE_SAMPLE_COUNT: ns,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(round(dt * 1e6)),
                    segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                })
                dst.trace[i] = data[:, i].astype(np.float32)
        finally:
            if src is not None:
                src.close()
        dst.bin.update(hns=ns, hdt=int(round(dt * 1e6)), format=5)
