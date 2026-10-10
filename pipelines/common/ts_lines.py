"""Prefix every stdin line (\\n or \\r terminated, e.g. tqdm) with a unix timestamp: '<t> <line>'.
Binary-safe: undecodable bytes are replaced instead of killing the logger (and with it the training)."""
import sys
import time


def main():
    stdin, out = sys.stdin.buffer, sys.stdout
    buffer = bytearray()
    while True:
        chunk = stdin.read1(65536)
        if not chunk:
            break
        for byte in chunk:
            if byte in (10, 13):
                if buffer.strip():
                    out.write(f"{time.time():.3f} {buffer.decode('utf-8', 'replace')}\n")
                buffer.clear()
            else:
                buffer.append(byte)
        out.flush()
    if buffer.strip():
        out.write(f"{time.time():.3f} {buffer.decode('utf-8', 'replace')}\n")
        out.flush()


if __name__ == "__main__":
    main()
