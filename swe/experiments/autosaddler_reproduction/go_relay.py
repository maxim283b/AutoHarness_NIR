"""Fixed-upstream byte relay; it cannot select a model or arbitrary destination."""
import os
import select
import socket
import socketserver


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        host = os.environ['AUTOHARNESS_RELAY_HOST']
        port = int(os.environ['AUTOHARNESS_RELAY_PORT'])
        with socket.create_connection((host,port),timeout=30) as upstream:
            upstream.settimeout(None)
            sockets = [self.request,upstream]
            while True:
                readable,_,_ = select.select(sockets,[],[],30)
                for source in readable:
                    data = source.recv(65536)
                    if not data:return
                    (upstream if source is self.request else self.request).sendall(data)


if __name__ == '__main__':
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    with Server(('0.0.0.0',8080),Relay) as server:server.serve_forever()
