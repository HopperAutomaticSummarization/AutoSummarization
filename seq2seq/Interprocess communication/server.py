#coding:utf8

from socket import socket, AF_INET, SOCK_STREAM

port = 50008
host = 'localhost'


def server():
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind(('',port))
    sock.listen(5)
    while True:
        conn,addr = sock.accept()
        data = conn.recv(1024)
        reply = 'server got:[%s]' % data
        conn.send(reply.encode())


