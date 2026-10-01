"""Tests for mux_controller.py — hex-protocol command generation."""

import pytest
from solarjv_analyzer.instruments.mux_controller import MuxController


class TestMuxController:
    """Unit tests for MuxController hex commands (no serial port)."""

    @pytest.fixture
    def mux(self):
        return MuxController(port="COM99")

    def test_select_channel_hex_generation(self, mux, monkeypatch):
        """select_channel() computes correct hex for each channel."""
        sent = []

        def fake_write(data):
            sent.append(data.hex().upper())

        monkeypatch.setattr(mux, "_send_hex", lambda hx: fake_write(bytes.fromhex(hx)))
        # Patch serial so connect doesn't fail and _send_hex works
        mux.ser = True  # fake connected state
        import serial
        monkeypatch.setattr(serial, "Serial", lambda **kw: None)

        for ch, expected_hex in [
            (1, "AA0100000000BB"),
            (2, "AA0101000000BB"),
            (3, "AA0102000000BB"),
            (4, "AA0103000000BB"),
            (5, "AA0104000000BB"),
            (6, "AA0105000000BB"),
        ]:
            sent.clear()
            # Bypass sleep and serial write
            mux._send_hex = lambda hx, s=sent: s.append(hx)
            mux.select_channel(ch)
            assert sent[0] == expected_hex, f"Ch {ch}: {sent[0]} != {expected_hex}"

    def test_deselect_channel_hex_generation(self, mux):
        """deselect_channel() uses the off-command hex pattern."""
        sent = []
        mux._send_hex = lambda hx, s=sent: s.append(hx)
        mux.deselect_channel(1)
        assert sent[0] == "AA0100000100BB"

        sent.clear()
        mux.deselect_channel(4)
        assert sent[0] == "AA0103000100BB"

    def test_send_hex_raises_when_disconnected(self, mux):
        with pytest.raises(RuntimeError, match="not connected"):
            mux._send_hex("AA0100000000BB")

    def test_close_calls_serial_close(self, mux):
        """close() calls the underlying serial port's close()."""
        closed = []

        class FakeSerial:
            is_open = True

            @staticmethod
            def close():
                closed.append(True)

        mux.ser = FakeSerial()
        mux.close()
        assert len(closed) == 1  # serial.close() was called
