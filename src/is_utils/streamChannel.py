from is_wire.core import Channel
import socket

class StreamChannel(Channel):
    def __init__(self, uri="amqp://guest:guest@10.10.2.211:30000", exchange="is"):
        super().__init__(uri=uri, exchange=exchange)

    def consume_last(self, timeout=None, return_dropped=False):
        dropped = 0
        try:
            msg = super().consume(timeout=timeout)
        except socket.timeout:
            return False

        while True:
            try:
                # will raise an exceptin when no message remained
                msg = super().consume(timeout=0.0)
                dropped += 1
            except socket.timeout:
                return (msg, dropped) if return_dropped else msg
