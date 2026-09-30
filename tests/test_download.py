"""只在服务器测试下载边界：CDN 忽略 Range 时不能发布错误分段。"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from parallel_download import fetch_part


@pytest.mark.parametrize("valid_range", [True, False])
def test_range_download_integrity(tmp_path, valid_range):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(206)
            self.send_header("Content-Length", "5")
            self.send_header("Content-Range", "bytes 0-4/5" if valid_range else "bytes 1-5/6")
            self.end_headers()
            self.wfile.write(b"hello")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/file"
        if valid_range:
            path = fetch_part(url, tmp_path, 0, 0, 4, threading.Event())
            assert path.read_bytes() == b"hello"
        else:
            with pytest.raises(ValueError, match="Content-Range"):
                fetch_part(url, tmp_path, 0, 0, 4, threading.Event())
            assert not (tmp_path / "part_00000").exists()
        assert not list(tmp_path.glob("*.headers"))
        assert not list(tmp_path.glob("*.incomplete"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cancel_before_network_request(tmp_path):
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="已取消"):
        fetch_part("http://127.0.0.1:1/file", tmp_path, 0, 0, 4, cancelled)
    assert not list(tmp_path.iterdir())
