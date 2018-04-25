#coding:utf8

from server import server
from client import client

if __name__ == '__main__':
    # from threading import Thread
    # sthread = Thread(target=server)
    # sthread.daemon = True
    # sthread.start()
    # for i in range(5):
    #     Thread(target=client, args=('client%s' % i,)).start()
    server()

