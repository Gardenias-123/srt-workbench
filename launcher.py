"""Launch the SRT/TopoSRT GUI as a local web app.

Dev:      python launcher.py
Headless: python launcher.py --headless
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser

from waitress import serve

from app import create_app


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    default_host = os.environ.get("HOST", "127.0.0.1")
    default_port = os.environ.get("PORT", "0")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=int(default_port) if str(default_port).isdigit() else 0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    port = args.port or free_port()
    url = f"http://{args.host}:{port}/"
    app = create_app()

    server_thread = threading.Thread(
        target=lambda: serve(app, host=args.host, port=port, threads=4),
        daemon=True,
    )
    server_thread.start()
    print(f"SRT GUI: {url}", flush=True)

    if args.headless:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        import tkinter as tk
    except Exception:
        print("Tkinter 不可用，按 Ctrl+C 退出。", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return

    root = tk.Tk()
    root.title("SRT / TopoSRT 工作台")
    root.geometry("460x170")
    root.resizable(False, False)
    tk.Label(root, text="SRT / TopoSRT 工作台", font=("Microsoft YaHei", 14, "bold")).pack(pady=(18, 4))
    tk.Label(root, text=url, fg="#2563eb", font=("Consolas", 11)).pack()
    tk.Label(root, text="关闭此窗口即退出程序", fg="#667085", font=("Microsoft YaHei", 9)).pack(pady=(6, 10))

    def open_browser():
        webbrowser.open(url)

    def quit_app():
        root.destroy()
        os._exit(0)

    frame = tk.Frame(root)
    frame.pack()
    tk.Button(frame, text="打开浏览器", command=open_browser, width=14).pack(side="left", padx=6)
    tk.Button(frame, text="退出", command=quit_app, width=10).pack(side="left", padx=6)
    root.protocol("WM_DELETE_WINDOW", quit_app)
    root.mainloop()


if __name__ == "__main__":
    multiprocessing_free = getattr(sys, "frozen", False)
    if multiprocessing_free:
        import multiprocessing

        multiprocessing.freeze_support()
    main()
