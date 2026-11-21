import errno
import socket
import ssl
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from struct import pack
from unittest.mock import Mock

import pytest

from amqp import Connection
from amqp import transport as transport_module
from amqp.method_framing import frame_handler
from amqp.serialization import dumps, loads
from amqp.transport import SSLTransport, TCPTransport


HEARTBEAT = b'\x08\x00\x00\x00\x00\x00\x00\xce'
BODY_FRAME = b'\x03\x00\x01\x00\x00\x00\x04ping\xce'


@pytest.fixture(params=[TCPTransport, SSLTransport], ids=['tcp', 'tls'])
def stream(request):
    reader, writer = socket.socketpair()
    try:
        if request.param is SSLTransport:
            certs = Path(__file__).resolve().parents[1] / 'certs'
            server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            server.load_cert_chain(certs / 'client_certificate.pem',
                                   certs / 'client_key.pem')
            client = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            client.check_hostname = False
            client.verify_mode = ssl.CERT_NONE
            writer = server.wrap_socket(
                writer, server_side=True, do_handshake_on_connect=False)
            reader = client.wrap_socket(
                reader, server_hostname='localhost',
                do_handshake_on_connect=False)
            reader.settimeout(5)
            writer.settimeout(5)
            with ThreadPoolExecutor(max_workers=1) as executor:
                handshake = executor.submit(writer.do_handshake)
                reader.do_handshake()
                handshake.result(timeout=5)
        reader.settimeout(None)
        writer.settimeout(5)
        transport = request.param('localhost')
        transport.sock = reader
        transport.connected = True
        transport._quick_recv = (reader.read if request.param is SSLTransport
                                 else reader.recv)
        yield transport, writer
    finally:
        reader.close()
        writer.close()


@contextmanager
def guard_nonblocking_read(transport):
    """Fail a retry after would-block instead of hanging the test suite."""
    recv = transport._quick_recv
    blocked = False

    def receive(size):
        nonlocal blocked
        assert not blocked, 'retried a nonblocking read after would-block'
        try:
            return recv(size)
        except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            blocked = True
            raise

    transport._quick_recv = receive
    try:
        yield
    finally:
        transport._quick_recv = recv


@pytest.mark.parametrize('wire,expected,split', [
    pytest.param(wire, expected, split, id=f'{name}-{split}')
    for name, wire, expected in [
        ('heartbeat', HEARTBEAT, (8, 0, b'')),
        ('body', BODY_FRAME, (3, 1, b'ping')),
    ]
    for split in range(len(wire))
])
@pytest.mark.parametrize('timeout', [0, 0.01])
@pytest.mark.parametrize('raise_on_initial_eintr', [False, True])
def test_partial_frame_resumes(stream, wire, expected, split, timeout,
                               raise_on_initial_eintr):
    transport, writer = stream
    transport.raise_on_initial_eintr = raise_on_initial_eintr
    writer.sendall(wire[:split])

    # Polling again without new bytes must also retain the frame unchanged.
    for _ in range(2):
        with guard_nonblocking_read(transport):
            with transport.having_timeout(timeout):
                with pytest.raises(socket.timeout):
                    transport.read_frame()
        assert transport._read_buffer == wire[:split]
        assert transport.connected
        assert transport.sock.gettimeout() is None

    writer.sendall(wire[split:] + HEARTBEAT)
    with guard_nonblocking_read(transport):
        with transport.having_timeout(0):
            assert transport.read_frame() == expected
            assert transport.read_frame() == (8, 0, b'')
    assert transport._read_buffer == b''


@pytest.mark.parametrize('split', [7, 8, 9, 10, 11])
def test_split_large_payload_resumes(stream, monkeypatch, split):
    transport, writer = stream
    # Exercise the two-read payload path without allocating gigabytes.
    monkeypatch.setattr(transport_module, 'SIGNED_INT_MAX', 2)
    writer.sendall(BODY_FRAME[:split])
    with guard_nonblocking_read(transport):
        with transport.having_timeout(0):
            with pytest.raises(socket.timeout):
                transport.read_frame()
    assert transport._read_buffer == BODY_FRAME[:split]
    writer.sendall(BODY_FRAME[split:])
    with transport.having_timeout(0):
        assert transport.read_frame() == (3, 1, b'ping')


@pytest.mark.parametrize('split', [0, 3, 7, 9, 11])
def test_distinct_ewouldblock_preserves_frame(stream, monkeypatch, split):
    transport, writer = stream
    # Windows uses a different value for EWOULDBLOCK and EAGAIN.
    monkeypatch.setattr(errno, 'EWOULDBLOCK', 10035)
    transport._read_buffer = BODY_FRAME[:split]
    recv = transport._quick_recv
    transport._quick_recv = Mock(side_effect=[
        OSError(10035, 'would block'),
        AssertionError('retried a nonblocking read after would-block'),
    ])
    with pytest.raises(socket.timeout):
        with transport.having_timeout(0):
            transport.read_frame()
    assert transport._read_buffer == BODY_FRAME[:split]
    assert transport.connected

    transport._quick_recv = recv
    writer.sendall(BODY_FRAME[split:])
    with transport.having_timeout(0):
        assert transport.read_frame() == (3, 1, b'ping')


@pytest.mark.parametrize('timeout', [None, 0, 0.1])
@pytest.mark.parametrize('initial', [False, True])
@pytest.mark.parametrize('raise_on_initial_eintr', [False, True])
def test_eintr_policy_is_preserved(stream, timeout, initial,
                                   raise_on_initial_eintr):
    transport, _ = stream
    transport.raise_on_initial_eintr = raise_on_initial_eintr
    transport._quick_recv = Mock(side_effect=[
        b'pi', InterruptedError(errno.EINTR, 'interrupted'), b'ng',
    ])
    with transport.having_timeout(timeout):
        if initial and raise_on_initial_eintr:
            with pytest.raises(socket.timeout):
                transport._read(4, initial=initial)
            assert transport._read_buffer == b'pi'
        assert transport._read(4, initial=initial) == b'ping'


@pytest.mark.parametrize('timeout', [None, 0.1])
def test_blocking_eagain_retry_is_preserved(stream, timeout):
    transport, _ = stream
    transport._quick_recv = Mock(side_effect=[
        b'pi', BlockingIOError(errno.EAGAIN, 'try again'), b'ng',
    ])
    with transport.having_timeout(timeout):
        assert transport._read(4) == b'ping'


@pytest.mark.parametrize('split', [0, 3, 7, 9, 11])
def test_peer_close_is_not_a_poll_timeout(stream, split):
    transport, writer = stream
    writer.sendall(BODY_FRAME[:split])
    writer.close()
    with transport.having_timeout(0):
        with pytest.raises(OSError) as caught:
            transport.read_frame()
    assert not isinstance(caught.value, socket.timeout)
    assert not transport.connected


@pytest.mark.parametrize('stream', [TCPTransport], indirect=True, ids=['tcp'])
def test_read_after_close_preserves_socket_error(stream):
    transport, _ = stream
    transport.close()
    with pytest.raises(OSError) as expected:
        transport._quick_recv(1)
    with pytest.raises(OSError) as caught:
        transport.read_frame()
    assert type(caught.value) is type(expected.value)
    assert caught.value.errno == expected.value.errno
    assert not transport.connected


@pytest.mark.parametrize('split', [0, 3, 7, 9, 11])
def test_fatal_socket_error_is_not_a_poll_timeout(stream, split):
    transport, _ = stream
    transport._read_buffer = BODY_FRAME[:split]
    error = OSError(errno.ECONNRESET, 'connection reset')
    transport._quick_recv = Mock(side_effect=error)
    with transport.having_timeout(0):
        with pytest.raises(OSError) as caught:
            transport.read_frame()
    assert caught.value is error
    assert not transport.connected


def test_drain_events_consumes_split_heartbeat(stream):
    transport, writer = stream
    connection = Connection()
    connection._transport = transport
    callbacks = []
    connection.on_inbound_frame = frame_handler(
        connection, lambda *args: callbacks.append(args))
    writer.sendall(HEARTBEAT[:7])
    with guard_nonblocking_read(transport):
        with pytest.raises(socket.timeout):
            connection.drain_events(timeout=0)
    assert connection.bytes_recv == 0
    writer.sendall(HEARTBEAT[7:])
    with guard_nonblocking_read(transport):
        with pytest.raises(socket.timeout):
            connection.drain_events(timeout=0)
    assert connection.bytes_recv == 1
    assert callbacks == []
    assert transport._read_buffer == b''


@pytest.mark.parametrize('stream', [SSLTransport], indirect=True, ids=['tls'])
@pytest.mark.parametrize('error_type,error_code', [
    (ssl.SSLWantReadError, ssl.SSL_ERROR_WANT_READ),
    (ssl.SSLWantWriteError, ssl.SSL_ERROR_WANT_WRITE),
    (ssl.SSLError, ssl.SSL_ERROR_WANT_READ),
    (ssl.SSLError, ssl.SSL_ERROR_WANT_WRITE),
], ids=['want-read', 'want-write', 'generic-want-read', 'generic-want-write'])
@pytest.mark.parametrize('split', [0, 3, 7, 9, 11])
def test_ssl_want_io_preserves_frame(stream, error_type, error_code, split):
    transport, writer = stream
    transport._read_buffer = BODY_FRAME[:split]
    recv = transport._quick_recv
    transport._quick_recv = Mock(side_effect=[
        error_type(error_code, 'The operation did not complete'),
        AssertionError('retried a nonblocking read after would-block'),
    ])
    with pytest.raises(socket.timeout):
        with transport.having_timeout(0):
            transport.read_frame()
    assert transport._read_buffer == BODY_FRAME[:split]
    assert transport.connected
    assert transport.sock.gettimeout() is None

    transport._quick_recv = recv
    writer.sendall(BODY_FRAME[split:])
    with transport.having_timeout(0):
        assert transport.read_frame() == (3, 1, b'ping')


def test_drain_events_resumes_multiframe_message(stream):
    transport, writer = stream
    connection = Connection()
    connection._transport = transport
    callbacks = []
    connection.on_inbound_frame = frame_handler(
        connection, lambda *args: callbacks.append(args))
    # Basic.Deliver followed by its content header and two body frames.
    args = dumps('sLbss', ('consumer', 1, False, '', 'queue'))
    method = pack('>BHIHH', 1, 1, len(args) + 4, 60, 60) + args + b'\xce'
    header = pack('>BHIHHQHB', 2, 1, 14, 60, 0, 8, 0, 0xce)
    last_body = b'\x03\x00\x01\x00\x00\x00\x04pong\xce'
    writer.sendall(method + header + BODY_FRAME + HEARTBEAT + last_body[:9])
    with guard_nonblocking_read(transport):
        with pytest.raises(socket.timeout):
            connection.drain_events(timeout=0)
    assert callbacks == []
    assert connection.bytes_recv == 4

    writer.sendall(last_body[9:])
    with guard_nonblocking_read(transport):
        connection.drain_events(timeout=0)
    assert len(callbacks) == 1
    channel, signature, args, message = callbacks[0]
    assert channel == 1
    assert signature == (60, 60)
    delivery, offset = loads('sLbss', args, 4)
    assert delivery == ['consumer', 1, False, '', 'queue']
    assert offset == len(args)
    assert message.body == b'pingpong'
    assert connection.bytes_recv == 5
    assert transport._read_buffer == b''
