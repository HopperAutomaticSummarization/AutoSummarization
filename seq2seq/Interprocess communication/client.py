#coding:utf8

import socket

port = 50008
host = socket.gethostbyname(socket.gethostname())

def client(name):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host,port))
    sock.send(name)
    reply = sock.recv(1024)
    sock.close()
    print('client got:[%s]' % reply)

# client('实际上，上周主管部门就和大唐打过招呼了，内部消息人士透露，国资委已经就李小琳任职问题和大唐进行沟通，但李小琳本人至今未报到。情况比较复杂，上述人士表示，目前还不敢完全确定，不排除后续还有变化。')
client('近日，有消息称七龙珠中的主角孙悟空或成东京奥运形象大使。有中国网友认为，如果此事成真，中国的孙悟空将被日本抢走。但六小龄童并不这样认为:毕竟那个孙悟空，并不是中国传统文化中的孙悟空。')
