import pandas as pd
import numpy as np

file = pd.read_csv('KDDTest.csv')
file = file.drop(['Difficulty'], axis=1)
protocol = np.array(file['Protocol-type'])
service = np.array(file['Service'])
flag = np.array(file['Flag'])
classe = np.array(file['Class'])
i = 0
for item in protocol:
    if item == 'tcp':
        protocol[i] = 1
    elif item == 'udp':
        protocol[i] = 5
    else:
        protocol[i] = 10
    i = i+1
i = 0
for item in service:
    if item == 'aol':
        service[i] = 1
    elif item == 'auth':
        service[i] = 2
    elif item == 'bgp':
        service[i] = 3
    elif item == 'courier':
        service[i] = 4
    elif item == 'csnet_ns':
        service[i] = 5
    elif item == 'ctf':
        service[i] = 6
    elif item == 'daytime':
        service[i] = 7
    elif item == 'discard':
        service[i] = 8
    elif item == 'domain':
        service[i] = 9
    elif item == 'domain_u':
        service[i] = 10
    elif item == 'echo':
        service[i] = 11
    elif item == 'eco_i':
        service[i] = 12
    elif item == 'ecr_i':
        service[i] = 13
    elif item == 'efs':
        service[i] = 14
    elif item == 'exec':
        service[i] = 15
    elif item == 'finger':
        service[i] = 16
    elif item == 'ftp':
        service[i] = 17
    elif item == 'ftp_data':
        service[i] = 18
    elif item == 'gopher':
        service[i] = 19
    elif item == 'harvest':
        service[i] = 20
    elif item == 'hostnames':
        service[i] = 21
    elif item == 'http':
        service[i] = 22
    elif item == 'http_2784':
        service[i] = 23
    elif item == 'http_443':
        service[i] = 24
    elif item == 'http_8001':
        service[i] = 25
    elif item == 'imap4':
        service[i] = 26
    elif item == 'IRC':
        service[i] = 27
    elif item == 'iso_tsap':
        service[i] = 28
    elif item == 'klogin':
        service[i] = 29
    elif item == 'kshell':
        service[i] = 30
    elif item == 'ldap':
        service[i] = 31
    elif item == 'link':
        service[i] = 32
    elif item == 'login':
        service[i] = 33
    elif item == 'mtp':
        service[i] = 34
    elif item == 'name':
        service[i] = 35
    elif item == 'netbios_dgm':
        service[i] = 36
    elif item == 'netbios_ns':
        service[i] = 37
    elif item == 'netbios_ssn':
        service[i] = 38
    elif item == 'netstat':
        service[i] = 39
    elif item == 'nnsp':
        service[i] = 40
    elif item == 'nntp':
        service[i] = 41
    elif item == 'ntp_u':
        service[i] = 42
    elif item == 'other':
        service[i] = 43
    elif item == 'pm_dump':
        service[i] = 44
    elif item == 'pop_2':
        service[i] = 45
    elif item == 'pop_3':
        service[i] = 46
    elif item == 'printer':
        service[i] = 47
    elif item == 'private':
        service[i] = 48
    elif item == 'red_i':
        service[i] = 49
    elif item == 'remote_job':
        service[i] = 50
    elif item == 'rje':
        service[i] = 51
    elif item == 'shell':
        service[i] = 52
    elif item == 'smtp':
        service[i] = 53
    elif item == 'sql_net':
        service[i] = 54
    elif item == 'ssh':
        service[i] = 55
    elif item == 'sunrpc':
        service[i] = 56
    elif item == 'supdup':
        service[i] = 57
    elif item == 'systat':
        service[i] = 58
    elif item == 'telnet':
        service[i] = 59
    elif item == 'tftp_u':
        service[i] = 60
    elif item == 'tim_i':
        service[i] = 61
    elif item == 'time':
        service[i] = 62
    elif item == 'urh_i':
        service[i] = 63
    elif item == 'uucp':
        service[i] = 64
    elif item == 'uucp_path':
        service[i] = 56
    elif item == 'vmnet':
        service[i] = 66
    elif item == 'whois':
        service[i] = 67
    elif item == 'X11':
        service[i] = 68
    else:
        service[i] = 69
    i = i+1
i = 0
for item in flag:
    if item == 'OTH':
        flag[i] = 1
    elif item == 'REJ':
        flag[i] = 2
    elif item == 'RSTO':
        flag[i] = 3
    elif item == 'RSTOS0':
        flag[i] = 4
    elif item == 'RSTR':
        flag[i] = 5
    elif item == 'S0':
        flag[i] = 6
    elif item == 'S1':
        flag[i] = 7
    elif item == 'S2':
        flag[i] = 8
    elif item == 'S3':
        flag[i] = 9
    elif item == 'SF':
        flag[i] = 10
    else:
        flag[i] = 11
    i = i+1

i = 0
for item in classe:
    if item == 'normal':
        classe[i] = 0
    else:
        classe[i] = 1
    i = i+1

file['Protocol-type'] = protocol
file['Service'] = service
file['Flag'] = flag
file['Class'] = classe

file.to_csv('EKDD_test.csv', index=False)
