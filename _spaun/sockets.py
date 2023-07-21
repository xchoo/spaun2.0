from __future__ import absolute_import

import logging
import socket
import sys

import nengo
from nengo.exceptions import ValidationError
import numpy as np

logger = logging.getLogger(__name__)


# FIXME close sockets when simulator is closed, remove SO_REUSEPORT
# Currently Nengo does not provide a mechanism for this, thus we allow to
# reuse ports currently to avoid problems with addresses already in use (that
# would especially occur in the GUI).

# TODO better handling of shuffled packets
# If packets get shuffled during transmission, we only keep the first packet
# with a future timestamp and drop all packets with an earlier timestamp if
# they arrive after that packet. Those might still be usable if the current
# simulation time does not exceed the timestamp of those packages. This could
# probably be solved with a priority queue (Python module heapq) to insert
# future packages.

# TODO IPv6 support?


class ConnectionTimeout(RuntimeError):
    pass


class _UDPSocket(object):
    def __init__(self, addr, dims, byte_order, timeout=None):
        self.addr = addr
        self.dims = dims
        if byte_order == "little":
            byte_order = "<"
        elif byte_order == "big":
            byte_order = ">"
        if byte_order not in "<>=":
            raise ValidationError("Must be one of '<', '>', '=', 'little', "
                                  "'big'.", attr="byte_order")
        self.byte_order = byte_order
        if np.isscalar(timeout):
            self.timeout = (timeout, timeout)
        else:
            self.timeout = timeout
        if self.timeout is not None:
            self.current_timeout = max(self.timeout)
        else:
            self.current_timeout = None

        self._buffer = np.empty(dims + 1, dtype="%sf8" % byte_order)
        self._buffer[0] = np.nan
        self._socket = None

    @property
    def t(self):
        return self._buffer[0]

    @property
    def x(self):
        return self._buffer[1:]

    @property
    def closed(self):
        return self._socket is None

    def open(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if sys.platform.startswith('bsd') or sys.platform.startswith('darwin'):
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        else:
            # Linux >= 3.9 has SO_REUSEPORT, but does load balancing for it.
            # We want all data to go the last opened socket.
            # More details:
            # https://stackoverflow.com/questions/14388706/socket-options-so-reuseaddr-and-so-reuseport-how-do-they-differ-do-they-mean-t?rq=1
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.timeout is not None:
            self.current_timeout = max(self.timeout)
        else:
            self.current_timeout = None

    def bind(self):
        self._socket.bind(self.addr)

    def recv(self, timeout):
        logger.debug("Waiting for packet with timeout %fs.", timeout)
        self._socket.settimeout(timeout)
        self._socket.recv_into(self._buffer.data)
        logger.debug("Received packet for t=%fs.", self.t)

    def recv_with_adaptive_timeout(self):
        if self.current_timeout is not None:
            logger.debug(
                "Waiting for packet with adaptive timeout "
                "(current value %fs.)", self.current_timeout)
        else:
            logger.debug("Waiting for packet (blocking).")
        self._socket.settimeout(self.current_timeout)
        try:
            self._socket.recv_into(self._buffer.data)
            if self.current_timeout is not None:
                self.current_timeout = max(
                    min(self.timeout), 0.9 * self.current_timeout)
        except socket.timeout:
            # TODO is a slow increase in timeout better than a complete reset?
            if self.current_timeout is not None:
                self.current_timeout = max(self.timeout)
            raise

    def send(self, t, x):
        self._buffer[0] = t
        self._buffer[1:] = x
        self._socket.sendto(self._buffer.tobytes(), self.addr)
        logger.debug("Send packet for t=%fs.", self.t)

    def close(self):
        if not self.closed:
            self._socket.close()
            self._socket = None


class SocketStep(object):
    r"""Handles the step for socket processes.

    One critical thing in this is to align local and remote timesteps if the
    timestep width (dt) differs. To figure out, the right thing to do we have
    to consider two cases:

    1. Remote dt is smaller or equal than local dt.
    2. Remote dt is larger than local dt.

    But first, some notation: the local timestamp of the current step is
    denoted with :math:`t`, the local timestep width with :math:`dt`. For the
    corresponding remote variable :math:`t'` and :math:`dt'` are used. The
    :math:`t'` is the value read from a remote

    It is also helpful to visualize the timeline with a little diagram:

    .. code-block:: none

          1    2    3    4    5    6    7    8    timestep indices
        [ | ][ | ][ | ][ | ][ | ][ | ][ | ][ | ]
        [     |     ][     |     ][     |     ][
              1            2            3         timestep indices

    Each timestep can be represented as an interval ``[ | ]``. The ``|``
    denotes the middle of the interval and corresponds to :math:`t` and
    :math:`t'` respectively. The width of each ``[ | ]`` corresponds to
    :math:`dt` (or :math:`dt'`).

    Let us consider the first case. The bottom row in the diagram denotes the
    local end in this case. Sending packets is simple: we can just send
    a packet in each timestep because the remote end would be able to process
    even more data.

    For receiving packets, we have multiple options as we are potentially
    getting more packets then timesteps. We could average over multiple
    packets, use the packet closest to the timestep interval mean, or use the
    first packet that falls into the local timestep interval. In the code here,
    we are using that last option because it does not require knowledge of the
    timestep between sent packages on the remote end (depending on the
    implementation it might just send a package every timestep or adjust the
    sending frequency to the local :math:`dt`).

    The logic described in text here, can be expressed as an inequality for
    when to use a packet: :math:`t - dt/2 <= t' < t + dt/2`. In the `recv`
    method this inequality is split up into two parts. The left part is handled
    by the while loop (because it is a while and not an if condition, the logic
    of the condition gets inverted). The right inequality is handled by the
    following if condition. Note, that we set :math:`dt' = dt` if
    :math:`dt' <= dt` and thus we can use :math:`dt'` instead of :math:`dt`
    which allows us to use exactly the same code for the second case discussed
    next.

    In the second case the top row in the diagram corresponds to the remote
    end.  When receiving data, each local timestep should use the remote value
    from the remote timestep with the largest overlap in the interval (because
    that value will be most representative for the given local timestep). So
    given the picture above, local timesteps 1, 2, 3 should use the remote
    value for timestep 1; 4, 5 the value for 2; 6, 7, 8 the value for 3. Thus,
    the first local timestep that overlaps more than 50% with the next remote
    timestep interval should receive a new packet. Expressed as an equation, if
    :math:`t' + dt'/2 < t` (where :math:`t'` is the last received timestamp),
    a new packet should be received. Note that this is equivalent to the left
    inequality obtained in the first case, so we don't need special handling
    for this case. Also, the right inequality applies. If the received value
    does not fulfil :math:`t' - dt'/2 < t` (where :math:`t'` is now the
    timestep of the newly received packet), it is a value that corresponds to
    timesteps that are still in the future and should not be used yet.

    When sending data, we could send a packet every timestep, but this would
    flood the remote end with packets that it does not use (at least currently
    where only a single value is used and no averaging is done). So, we want
    to send the next packet at the :math:`t` closest to :math:`t' + dt'` (where
    :math:`t'` is the timestamp of the last sent packet). Expressed as an
    equation a packet should be send when :math:`t' + dt' <= t + dt/2`.
    """

    def __init__(self, dt, send=None, recv=None,
                 remote_dt=None, connection_timeout=None,
                 loss_limit=None, ignore_timestamp=False):
        self.send_socket = send
        self.recv_socket = recv
        self.remote_dt = remote_dt
        self.connection_timeout = connection_timeout
        self.loss_limit = loss_limit
        self.ignore_timestamp = ignore_timestamp

        self.dt = dt
        if remote_dt is None:
            remote_dt = dt
        # Cannot run faster than local dt
        self.remote_dt = max(remote_dt, dt)

        self.n_lost = 0

        # State used by the step function
        self.value = np.zeros(0 if self.recv_socket is None
                              else self.recv_socket.dims)

    def __call__(self, t, x=None):
        """The step function run on each timestep.

        When both sending and receiving, the sending frequency is
        regulated by comparing the local and remote time steps. Information
        is sent when the current local timestep is closer to the remote
        time step than the next local timestep.
        """
        if t <= 0.:  # Nengo calling this function to figure out output size
            return self.value

        # Send must happen before receive to avoid deadlock situations, i.e.
        # if both ends tried to receive first, both would block. Even with
        # a timeout, the timestamps would not align to the expected timestamps
        # anymore.
        if self.send_socket is not None:
            assert x is not None, "A sender must receive input"
            self.send(t, x)
        if self.recv_socket is not None and (
                self.loss_limit is None or self.n_lost <= self.loss_limit):
            try:
                self.recv(t)
                self.n_lost = 0
            except socket.timeout:  # packet lost
                logger.info("No packet received for t=%fs.", t)
                self.n_lost += 1
        return self.value

    def __del__(self):
        self.close()

    def close(self):
        if self.send_socket is not None:
            self.send_socket.close()
        if self.recv_socket is not None:
            self.recv_socket.close()

    def recv(self, t):
        if self.ignore_timestamp:
            self.recv_socket.recv_with_adaptive_timeout()
            self._update_value()
            return

        # Receive initial packet
        if np.isnan(self.recv_socket.t):
            try:
                self.recv_socket.recv(self.connection_timeout)
            except socket.timeout:
                raise ConnectionTimeout(
                    "Did not receive initial packet within connection "
                    "timeout.")
            self._update_value()

        # Wait for packet that is not timestamped in the past
        # (also skips receiving if we do not expect a new remote package yet)
        while self.recv_socket.t < t - self.remote_dt / 2.:
            self.recv_socket.recv_with_adaptive_timeout()

        # Use value if not in the future
        if self.recv_socket.t < t + self.remote_dt / 2.:
            self._update_value()

    def _update_value(self):
        # Value needs to be copied, otherwise it might be overwritten
        # prematurely by a packet for a future timestep.
        self.value = np.array(self.recv_socket.x)

    def send(self, t, x):
        # Calculate if it is time to send the next packet.
        # Ideal time to send is the last sent time + remote_dt, and we
        # want to find out if current or next local time step is closest.
        if (np.isnan(self.send_socket.t) or
                (t + self.dt / 2.) >= (self.send_socket.t + self.remote_dt)):
            self.send_socket.send(t, x)


class UDPReceiveSocket(nengo.Process):
    """A process for receiving data from a UDP socket in a Nengo model.

    The *size_out* attributes of the `nengo.Node` using this
    process determines the dimensions of the received data.

    The incoming UDP packets are expected to start with the timestep followed
    by the values for that timestep. Each value should be encoded as 8-byte
    floating point number.

    A packet will be used if its timestep is within within a window with the
    width of *remote_dt* centered around the current time.

    Parameters
    ----------
    listen_addr : tuple
        A tuple *(listen_interface, port)* denoting the local address to listen
        on for incoming data.
    remote_dt : float, optional (Default: None)
        The timestep of the remote simulation. Attempts to send and receive
        data will be throttled to match this value if it exceeds the local
        *dt*. If not given, it is assumed that the remote *dt* matches the
        local *dt* (which is determined automatically).
    connection_timeout : float, optional (Default: 300.)
        Initial timeout when waiting to receive the initial package
        establishing the connection.
    recv_timeout : 2-tuple or float or None, optional (Default: 0.1)
        Timout for socket receive operations in each timestep. If *None*, there
        is no timeout (block until package is received). A float denotes a
        fixed timeout. A 2-tuple gives a minimum and maximum timeout and the
        timeout will be adjusted adaptively between these two values.
    loss_limit: float, optional (Default: None)
        If not *None*, the maximum number of consecutive timeouts on receive
        attempts before no further attempts are made and the last received
        value will be used for the rest of the simulation.
    byte_order : str, optional (Default: '=')
        Specify 'big' or 'little' endian data format.
        Possible values: 'big', '>', 'little', '<', '='.
        '=' uses the system default.

    Examples
    --------
    To receive data on a machine with IP address 10.10.21.1,
    we add the following socket to the model::

        socket_recv = UDPReceiveSocket(('10.10.21.1', 5001))
        node_recv = nengo.Node(socket_recv, size_out=recv_dims)

    Other Nengo model elements can then be connected to the node.
    """
    def __init__(self, listen_addr, remote_dt=None,
                 connection_timeout=300., recv_timeout=0.1, loss_limit=0,
                 byte_order='='):
        super(UDPReceiveSocket, self).__init__(default_size_in=0)
        self.listen_addr = listen_addr
        self.remote_dt = remote_dt
        self.connection_timeout = connection_timeout
        self.recv_timeout = recv_timeout
        self.loss_limit = loss_limit
        self.byte_order = byte_order

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert len(shape_out) == 1
        recv = _UDPSocket(
            self.listen_addr, shape_out[0], self.byte_order,
            timeout=self.recv_timeout)
        recv.open()
        recv.bind()
        return SocketStep(
            dt=dt, recv=recv, remote_dt=self.remote_dt,
            connection_timeout=self.connection_timeout,
            loss_limit=self.loss_limit)


class UDPSendSocket(nengo.Process):
    """A process for sending data from a Nengo model through a UDP socket.

    The *size_in* attributes of the `nengo.Node` using this
    process determines the dimensions of the sent data.

    The outgoing UDP packets are will start with the timestep followed
    by the values for that timestep. Each value will be encoded as 8-byte
    floating point number.

    Parameters
    ----------
    remote_addr : tuple
        A tuple *(host, port)* denoting the remote address to send data to
    remote_dt : float, optional (Default: None)
        The timestep of the remote simulation. Attempts to send and receive
        data will be throttled to match this value if it exceeds the local
        *dt*. If not given, it is assumed that the remote *dt* matches the
        local *dt* (which is determined automatically).
    byte_order : str, optional (Default: '=')
        Specify 'big' or 'little' endian data format.
        Possible values: 'big', '>', 'little', '<', '='.
        '=' uses the system default.

    Examples
    --------
    To send data from a model to a machine with IP address 10.10.21.25,
    we add the following socket to the model::

        socket_send = UDPSendSocket(('10.10.21.25', 5002))
        node_send = nengo.Node(socket_send, size_in=send_dims)

    Other Nengo model elements can then be connected to the node.
    """
    def __init__(self, remote_addr, remote_dt=None, byte_order="="):
        super(UDPSendSocket, self).__init__(default_size_out=0)
        self.remote_addr = remote_addr
        self.remote_dt = remote_dt
        self.byte_order = byte_order

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert len(shape_in) == 1
        send = _UDPSocket(self.remote_addr, shape_in[0], self.byte_order)
        send.open()
        return SocketStep(dt=dt, send=send, remote_dt=self.remote_dt)


class UDPSendReceiveSocket(nengo.Process):
    """A process for UDP communication to and from a Nengo model.

    The *size_in* and *size_out* attributes of the `nengo.Node` using this
    process determines the dimensions of the sent and received data.

    The incoming UDP packets are expected to start with the timestep followed
    by the values for that timestep. Each value should be encoded as 8-byte
    floating point number. The outgoing packets follow the same format.

    A received packet will be used if its timestep is within within a window
    with the width of *remote_dt* centered around the current time.

    Parameters
    ----------
    listen_addr : tuple
        A tuple *(listen_interface, port)* denoting the local address to listen
        on for incoming data.
    remote_addr : tuple
        A tuple *(host, port)* denoting the remote address to send data to
    remote_dt : float, optional (Default: None)
        The timestep of the remote simulation. Attempts to send and receive
        data will be throttled to match this value if it exceeds the local
        *dt*. If not given, it is assumed that the remote *dt* matches the
        local *dt* (which is determined automatically).
    ignore_timestamp : boolean, optional (Default: False)
        If True, uses the most recently received value from the recv socket,
        even if that value comes at an earlier or later timestep.
    connection_timeout : float, optional (Default: 300.)
        Initial timeout when waiting to receive the initial package
        establishing the connection.
    recv_timeout : 2-tuple or float or None, optional (Default: 0.1)
        Timout for socket receive operations in each timestep. If *None*, there
        is no timeout (block until package is received). A float denotes a
        fixed timeout. A 2-tuple gives a minimum and maximum timeout and the
        timeout will be adjusted adaptively between these two values.
    loss_limit: float, optional (Default: None)
        If not *None*, the maximum number of consecutive timeouts on receive
        attempts before no further attempts are made and the last received
        value will be used for the rest of the simulation.
    byte_order : str, optional (Default: '=')
        Specify 'big' or 'little' endian data format.
        Possible values: 'big', '>', 'little', '<', '='.
        '=' uses the system default.

    Examples
    --------
    To communicate between two models in send and receive mode over a network,
    one running on machine A with IP address 10.10.21.1 and one running on
    machine B, with IP address 10.10.21.25, we add the following socket to the
    model on machine A::

        socket_send_recv_A = UDPSendReceiveSocket(
            listen_addr=('10.10.21.1', 5001),
            remote_addr=('10.10.21.25', 5002))
        node_send_recv_A = nengo.Node(
            socket_send_recv_A,
            size_in=A_output_dims,
            size_out=B_output_dims)

    and the following socket on machine B::

        socket_send_recv_B = UDPSendReceiveSocket(
            listen_addr=('10.10.21.25', 5002),
            remote_addr=('10.10.21.1', 5001))
        node_send_recv_B = nengo.Node(
            socket_send_recv_B,
            size_in=B_output_dims,  # input to this node is data to send
            size_out=A_output_dims)  # output from this node is data received

    The nodes can then be connected to other Nengo model elements.
    """
    def __init__(
            self, listen_addr, remote_addr, remote_dt=None,
            ignore_timestamp=False, connection_timeout=300.,
            recv_timeout=0.1, loss_limit=None, byte_order='='):
        super(UDPSendReceiveSocket, self).__init__()
        self.listen_addr = listen_addr
        self.remote_addr = remote_addr
        self.remote_dt = remote_dt
        self.ignore_timestamp = ignore_timestamp
        self.connection_timeout = connection_timeout
        self.recv_timeout = recv_timeout
        self.loss_limit = loss_limit
        self.byte_order = byte_order


    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert len(shape_in) == 1
        assert len(shape_out) == 1
        recv = _UDPSocket(
            self.listen_addr, shape_out[0], self.byte_order,
            timeout=self.recv_timeout)
        recv.open()
        recv.bind()
        send = _UDPSocket(self.remote_addr, shape_in[0], self.byte_order)
        send.open()
        return SocketStep(
            dt=dt,
            send=send, recv=recv,
            ignore_timestamp=self.ignore_timestamp,
            remote_dt=self.remote_dt,
            connection_timeout=self.connection_timeout,
            loss_limit=self.loss_limit)
