"""
src/features.py
---------------
Extracts 2,381 numerical features from a Windows PE (.exe / .dll) binary file
using the pefile library — matching the EMBER 2018 v2 feature schema as closely
as possible without the LIEF dependency.

Feature groups:
  1.  General file info          (10 features)
  2.  Header fields              (62 features)
  3.  Section table stats        (50 features × 5 stats = 250 features)
  4.  Imports (DLL hashing)      (256 + 1024 features)
  5.  Exports                    (128 features)
  6.  Byte histogram             (256 features)
  7.  Byte-entropy histogram     (256 features)
  8.  String features            (104 features)
  9.  Data directory flags       (15 features)
  ──────────────────────────────────────────
  Total padded/clipped to        2,381 features
"""

import re
import math
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

TARGET_DIM = 2381   # Must match EMBER 2018 v2 vector length

try:
    import pefile
    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False
    log.warning("pefile not installed — feature extraction will return zeros.")


# ── Helper utilities ──────────────────────────────────────────────────────────

def _safe_entropy(data: bytes) -> float:
    """Shannon entropy of a byte sequence, in bits."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _byte_histogram(data: bytes) -> np.ndarray:
    """Normalised 256-bin byte frequency histogram."""
    h = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float32)
    total = h.sum()
    return h / total if total > 0 else h


def _byte_entropy_histogram(data: bytes, window: int = 2048, step: int = 1024) -> np.ndarray:
    """
    16×16 joint histogram of (byte_value // 16, entropy_bin) over sliding windows.
    Flattened to 256 values and normalised.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    hist = np.zeros(256, dtype=np.float32)
    if len(arr) < window:
        return hist
    for start in range(0, len(arr) - window + 1, step):
        chunk = arr[start:start + window]
        counts = np.bincount(chunk, minlength=256)
        p = counts / window
        p_nz = p[p > 0]
        ent = float(-np.sum(p_nz * np.log2(p_nz)))
        ent_bin = min(int(ent / 8.0 * 16), 15)
        for byte_val, cnt in enumerate(counts):
            if cnt > 0:
                byte_bin = byte_val // 16
                hist[byte_bin * 16 + ent_bin] += cnt
    total = hist.sum()
    return hist / total if total > 0 else hist


def _hash_features(items, n_bins: int) -> np.ndarray:
    """Feature-hashing trick: map string items into an n-bin float array."""
    vec = np.zeros(n_bins, dtype=np.float32)
    for item in items:
        idx = int(hashlib.md5(item.encode("utf-8", errors="replace")).hexdigest(), 16) % n_bins
        vec[idx] += 1.0
    return vec


def _extract_strings(data: bytes):
    """Extract printable ASCII substrings (length ≥ 5)."""
    pattern = rb"[\x20-\x7e]{5,}"
    return [m.group().decode("ascii", errors="replace") for m in re.finditer(pattern, data)]


# ── Main extraction function ──────────────────────────────────────────────────

def extract_features(file_path: str) -> np.ndarray:
    """
    Parse a PE file and return a float32 numpy array of length TARGET_DIM.
    Returns a zero vector if parsing fails.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the PE binary.

    Returns
    -------
    np.ndarray  shape (TARGET_DIM,)
    """
    if not PEFILE_AVAILABLE:
        return np.zeros(TARGET_DIM, dtype=np.float32)

    try:
        raw = Path(file_path).read_bytes()
        pe  = pefile.PE(file_path, fast_load=False)
    except Exception as exc:
        log.warning(f"pefile failed on {file_path}: {exc}")
        return np.zeros(TARGET_DIM, dtype=np.float32)

    parts = []

    # ── 1. General info (10) ──────────────────────────────────────────────────
    general = np.zeros(10, dtype=np.float32)
    general[0] = len(raw)
    general[1] = float(hasattr(pe, "VS_VERSIONINFO") and bool(pe.VS_VERSIONINFO))
    general[2] = float(bool(pe.DIRECTORY_ENTRY_DEBUG if hasattr(pe, "DIRECTORY_ENTRY_DEBUG") else 0))
    general[3] = float(bool(pe.OPTIONAL_HEADER.DATA_DIRECTORY[4].VirtualAddress) if hasattr(pe, "OPTIONAL_HEADER") else 0)  # tls
    try:
        general[4] = float(pe.FILE_HEADER.NumberOfSymbols)
        general[5] = float(pe.FILE_HEADER.NumberOfSections)
        general[6] = float(pe.OPTIONAL_HEADER.MajorLinkerVersion)
        general[7] = float(pe.OPTIONAL_HEADER.MinorLinkerVersion)
        general[8] = float(pe.FILE_HEADER.TimeDateStamp)
        general[9] = float(pe.OPTIONAL_HEADER.SizeOfCode)
    except Exception:
        pass
    parts.append(general)

    # ── 2. Header fields (62) ─────────────────────────────────────────────────
    header = np.zeros(62, dtype=np.float32)
    try:
        oh = pe.OPTIONAL_HEADER
        fh = pe.FILE_HEADER
        fields_oh = [
            "MajorImageVersion", "MinorImageVersion",
            "MajorOperatingSystemVersion", "MinorOperatingSystemVersion",
            "MajorSubsystemVersion", "MinorSubsystemVersion",
            "SizeOfStackReserve", "SizeOfStackCommit",
            "SizeOfHeapReserve", "SizeOfHeapCommit",
            "SizeOfImage", "SizeOfHeaders",
            "CheckSum", "Subsystem", "DllCharacteristics",
        ]
        for i, f in enumerate(fields_oh[:min(len(fields_oh), 62)]):
            header[i] = float(getattr(oh, f, 0))
        header[15] = float(fh.Machine)
        header[16] = float(fh.Characteristics)
        header[17] = float(fh.SizeOfOptionalHeader)
    except Exception:
        pass
    parts.append(header)

    # ── 3. Sections (250) ────────────────────────────────────────────────────
    section_feat = np.zeros(250, dtype=np.float32)
    try:
        sections = pe.sections[:50] if pe.sections else []
        for i, sec in enumerate(sections):
            base = i * 5
            data = sec.get_data()
            section_feat[base + 0] = float(sec.SizeOfRawData)
            section_feat[base + 1] = float(sec.Misc_VirtualSize)
            section_feat[base + 2] = _safe_entropy(data)
            section_feat[base + 3] = float(sec.Characteristics)
            section_feat[base + 4] = float(len(data))
    except Exception:
        pass
    parts.append(section_feat)

    # ── 4. Imports: DLL hashing (256) + DLL:Func hashing (1024) ──────────────
    dll_names, func_pairs = [], []
    try:
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll = entry.dll.decode("utf-8", errors="replace").lower()
                dll_names.append(dll)
                for imp in entry.imports:
                    if imp.name:
                        fn = imp.name.decode("utf-8", errors="replace").lower()
                        func_pairs.append(f"{dll}:{fn}")
    except Exception:
        pass
    parts.append(_hash_features(dll_names, 256))
    parts.append(_hash_features(func_pairs, 1024))

    # ── 5. Exports (128) ─────────────────────────────────────────────────────
    export_names = []
    try:
        if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if exp.name:
                    export_names.append(exp.name.decode("utf-8", errors="replace").lower())
    except Exception:
        pass
    parts.append(_hash_features(export_names, 128))

    # ── 6. Byte histogram (256) ───────────────────────────────────────────────
    parts.append(_byte_histogram(raw))

    # ── 7. Byte-entropy histogram (256) ──────────────────────────────────────
    parts.append(_byte_entropy_histogram(raw))

    # ── 8. String features (104) ─────────────────────────────────────────────
    strings = _extract_strings(raw)
    str_feat = np.zeros(104, dtype=np.float32)
    if strings:
        lengths = [len(s) for s in strings]
        str_feat[0] = len(strings)
        str_feat[1] = np.mean(lengths)
        str_feat[2] = np.std(lengths)
        str_feat[3] = sum(1 for s in strings if s.startswith(("http://", "https://")))
        str_feat[4] = sum(1 for s in strings if s.startswith(("C:\\", "c:\\")))
        str_feat[5] = sum(1 for s in strings if "HKEY_" in s)
        all_str = " ".join(strings)
        str_feat[6] = _safe_entropy(all_str.encode())
        str_feat[7] = sum(1 for s in strings if ".exe" in s.lower())
        str_feat[8] = sum(1 for s in strings if ".dll" in s.lower())
        str_feat[9] = sum(1 for s in strings if len(s) > 50)
        # Length histogram (bins: 0-4, 5-9, 10-19, 20-49, 50+) → 5 bins per
        for j, length in enumerate(sorted(lengths)[:94]):  # fill remaining
            str_feat[10 + j] = float(length)
    parts.append(str_feat)

    # ── 9. Data directory flags (15) ─────────────────────────────────────────
    dd_feat = np.zeros(15, dtype=np.float32)
    try:
        for i, dd in enumerate(pe.OPTIONAL_HEADER.DATA_DIRECTORY[:15]):
            dd_feat[i] = float(dd.VirtualAddress > 0)
    except Exception:
        pass
    parts.append(dd_feat)

    # ── Assemble and resize to TARGET_DIM ─────────────────────────────────────
    vec = np.concatenate(parts).astype(np.float32)
    if len(vec) < TARGET_DIM:
        vec = np.pad(vec, (0, TARGET_DIM - len(vec)))
    else:
        vec = vec[:TARGET_DIM]

    # Replace NaN / Inf
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def extract_features_dict(file_path: str) -> Dict[str, Any]:
    """
    Return a human-readable dictionary of key PE metadata for display in the UI.
    Separate from the ML feature vector.
    Includes extracted text features (DLLs, APIs, strings) for SHAP explainability.
    """
    info: Dict[str, Any] = {
        "file_size_bytes": 0,
        "num_sections": 0,
        "imports": [],
        "exports": [],
        "has_tls": False,
        "has_debug": False,
        "has_resources": False,
        "timestamp": None,
        "machine_type": "Unknown",
        "entry_point": 0,
        "suspicious_apis": [],
        "extracted_strings": [],  # For SHAP text feature explanations
        "dll_names": [],  # List of imported DLLs
        "suspicious_indicators": [],  # URLs, registry keys, file paths
    }

    SUSPICIOUS = {
        "VirtualAlloc", "VirtualAllocEx", "CreateRemoteThread",
        "WriteProcessMemory", "ReadProcessMemory", "OpenProcess",
        "NtUnmapViewOfSection", "SetWindowsHookEx", "GetAsyncKeyState",
        "URLDownloadToFile", "ShellExecute", "WinExec",
        "IsDebuggerPresent", "CheckRemoteDebuggerPresent",
    }

    if not PEFILE_AVAILABLE:
        return info

    try:
        raw = Path(file_path).read_bytes()
        pe  = pefile.PE(file_path, fast_load=False)
        info["file_size_bytes"] = len(raw)
        info["num_sections"] = len(pe.sections)
        info["entry_point"] = hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint)
        info["timestamp"] = pe.FILE_HEADER.TimeDateStamp

        machine_map = {0x14c: "x86 (i386)", 0x8664: "x64 (AMD64)", 0x1c0: "ARM"}
        info["machine_type"] = machine_map.get(pe.FILE_HEADER.Machine, hex(pe.FILE_HEADER.Machine))

        info["has_tls"]       = bool(pe.OPTIONAL_HEADER.DATA_DIRECTORY[9].VirtualAddress)
        info["has_debug"]     = hasattr(pe, "DIRECTORY_ENTRY_DEBUG")
        info["has_resources"] = hasattr(pe, "DIRECTORY_ENTRY_RESOURCE")

        suspicious_found = []
        dll_names = []
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll = entry.dll.decode("utf-8", errors="replace")
                dll_names.append(dll)
                fns = []
                for imp in entry.imports:
                    if imp.name:
                        fn = imp.name.decode("utf-8", errors="replace")
                        fns.append(fn)
                        if fn in SUSPICIOUS:
                            suspicious_found.append(fn)
                info["imports"].append({"dll": dll, "functions": fns})
        
        info["dll_names"] = dll_names
        info["suspicious_apis"] = suspicious_found

        if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if exp.name:
                    info["exports"].append(exp.name.decode("utf-8", errors="replace"))
        
        # Extract text strings for SHAP explanations
        extracted = _extract_strings(raw)
        info["extracted_strings"] = extracted[:50]  # Limit for readability
        
        # Categorize suspicious indicators
        indicators = []
        for s in extracted:
            if s.startswith(("http://", "https://")):
                indicators.append(("URL", s))
            elif s.startswith(("C:\\", "c:\\")):
                indicators.append(("File Path", s))
            elif "HKEY_" in s:
                indicators.append(("Registry", s))
        info["suspicious_indicators"] = indicators[:20]  # Limit for readability

    except Exception as exc:
        log.warning(f"Metadata extraction failed: {exc}")

    return info
