#!/usr/bin/env python3
# repair_csv_quotes.py
import sys, csv

def balanced(buf: str, quotechar='"') -> bool:
    """Return True if quotes are balanced in the current buffer."""
    inq = False
    i = 0
    n = len(buf)
    while i < n:
        ch = buf[i]
        if ch == quotechar:
            # handle doubled quotes inside a quoted field
            if inq and i + 1 < n and buf[i + 1] == quotechar:
                i += 2
                continue
            inq = not inq
        i += 1
    return not inq

def main(inp, outp):
    n_in = n_out = 0
    dropped_tail = False
    with open(inp, "r", encoding="utf-8", errors="ignore") as fin, \
         open(outp, "w", encoding="utf-8", newline="") as fout:
        buf = ""
        for line in fin:
            n_in += 1
            # normalize Windows newlines early to avoid false-positives
            line = line.replace("\r\n", "\n").replace("\r", "\n")
            buf += line
            if balanced(buf):
                if not buf.endswith("\n"):
                    buf += "\n"
                # normalize to LF-only
                fout.write(buf.replace("\r\n", "\n").replace("\r", "\n"))
                buf = ""
                n_out += 1
        if buf.strip():
            # last record was still open; drop it (truncated)
            dropped_tail = True

    print(f"[repair] lines_in={n_in}, records_out={n_out}, dropped_tail={dropped_tail}")
    if dropped_tail:
        print("⚠️  The last record was truncated and has been dropped.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python repair_csv_quotes.py <input.csv> <output.csv>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
