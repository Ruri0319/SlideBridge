from __future__ import annotations

import os
import sys
import time

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


def get_process_memory_mb() -> float:
    if psutil is not None:
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            pass

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return counters.WorkingSetSize / (1024 * 1024)
        except Exception:
            return 0.0
        return 0.0

    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024
    except Exception:
        return 0.0


class ProcessMetricsSampler:
    """Sample current process memory and CPU utilization without blocking."""

    def __init__(self):
        self._last_wall = time.perf_counter()
        self._last_cpu = time.process_time()
        self._cpu_count = max(1, os.cpu_count() or 1)
        self._process = None
        if psutil is not None:
            try:
                self._process = psutil.Process()
                self._process.cpu_percent(interval=None)
            except Exception:
                self._process = None

    def sample(self) -> tuple[float, float]:
        now_wall = time.perf_counter()
        now_cpu = time.process_time()
        elapsed_wall = max(0.001, now_wall - self._last_wall)
        elapsed_cpu = max(0.0, now_cpu - self._last_cpu)
        self._last_wall = now_wall
        self._last_cpu = now_cpu

        cpu_percent = (elapsed_cpu / elapsed_wall) * 100.0 / self._cpu_count
        if self._process is not None:
            try:
                cpu_percent = self._process.cpu_percent(interval=None) / self._cpu_count
            except Exception:
                self._process = None

        return get_process_memory_mb(), min(100.0, max(0.0, cpu_percent))


class PerfTracker:
    def __init__(self):
        self._started = time.perf_counter()
        self.peak_memory_mb = get_process_memory_mb()
        self._cpu_start = time.process_time()

    def sample(self) -> None:
        self.peak_memory_mb = max(self.peak_memory_mb, get_process_memory_mb())

    def average_cpu_percent(self) -> float:
        wall = max(0.001, time.perf_counter() - self._started)
        cpu = max(0.0, time.process_time() - self._cpu_start)
        cpu_count = max(1, os.cpu_count() or 1)
        return min(100.0, (cpu / wall) * 100.0 / cpu_count)
