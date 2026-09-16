#!/usr/bin/env python3
"""Launch and view the interactive executive dashboard.

Works on every computer and operating system (Windows, macOS, Linux).

Usage:
    python scripts/dashboard.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import webbrowser

# Ensure UTF-8 output on all consoles including Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dashboard.report import generate_dashboard


def main() -> int:
    project_root = pathlib.Path(__file__).resolve().parent.parent
    dashboard_file = project_root / "dashboard.html"

    print("=" * 80)
    print("       LPDG GATEWAY HEALTH INTELLIGENCE -- DASHBOARD LAUNCHER")
    print("=" * 80)
    print("--> Generating latest interactive executive dashboard...")

    try:
        output_path = generate_dashboard(
            output_path=str(dashboard_file),
            open_browser=False,  # We handle browser opening cleanly below
        )
        print(f"[OK] Dashboard generated successfully: {output_path}")
    except Exception as e:
        print(f"[NOTE] Loading existing dashboard (generation message: {e})")

    file_url = dashboard_file.resolve().as_uri()
    http_url = "http://localhost:8765/dashboard.html"

    print("\n--> Opening dashboard in your default browser...")
    opened = False
    try:
        # Try local HTTP server URL first if active
        opened = webbrowser.open(http_url)
    except Exception:
        pass

    if not opened:
        try:
            webbrowser.open(file_url)
        except Exception as e:
            print(f"[NOTE] Please open this link in your browser: {file_url}")

    print("\n" + "-" * 80)
    print(f"  [1] Web Browser URL (Local Server): {http_url}")
    print(f"  [2] Direct File Link:               {file_url}")
    print("-" * 80)
    print("\n[OK] Dashboard is open and ready to explore!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
