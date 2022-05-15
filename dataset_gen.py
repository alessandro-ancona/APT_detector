import numpy as np
import pandas as pd
import random

random.seed(123)

columns = ['alert_type', 'timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'infected_host', 'scanned_host']

# alert_types = ['disguised_exe_alert' 1,
#                         'hash_alert' 2,
#                       'domain_alert' 3,
#                           'ip_alert' 4,
#                          'ssl_alert' 5,
#                  'domain_flux_alert' 6,
#                         'scan_alert' 7,
#                          'tor_alert' 8]

random_alerts = pd.DataFrame(np.zeros((10000, 8)), columns=columns)

# RANDOM ALERT GENERATION ----------------------------------------------------------------------------------------------

for i in range(10000):
    alert_type = random.randint(1, 8)
    timestamp = random.randint(1, 1296000)
    src_ip = random.randint(1, 500)  # in campus network
    src_port = random.randint(49152, 65535)
    scanned_host = 0
    if alert_type == 1:  #'disguised_exe_alert'
        prob = random.randint(0, 100)
        if prob <= 70:
            dst_ip = random.randint(1001, 10000)  # fuori organizzazione
        else:
            dst_ip = random.randint(1, 500)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 80
        else:
            dst_port = random.randint(1025, 65535)
    elif alert_type == 2:  #'hash_alert'
        prob = random.randint(0, 100)
        if prob <= 70:
            dst_ip = random.randint(1001, 10000) # esterno all'organizzazione
        else:
            dst_ip = random.randint(1, 500)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 80
        else:
            dst_port = random.randint(1025, 65535)
    elif alert_type == 3:  #'domain_alert'
        prob = random.randint(0, 100)
        if prob <= 70:
            dst_ip = random.randint(1, 500) # nell organizzazione
        else:
            dst_ip = random.randint(1001, 10000)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 53
        else:
            dst_port = random.randint(1025, 65535)
    elif alert_type == 4:  #'ip_alert'
        prob = random.randint(0, 100)
        if prob <= 70:
            dst_ip = random.randint(1001, 1100) #nella ip blacklist
        else:
            dst_ip = random.randint(1101, 10000)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 443
        else:
            dst_port = random.randint(1025, 65535)
    elif alert_type == 5:  #'ssl_alert'
        prob = random.randint(0, 100)
        if prob <= 70:
            dst_ip = random.randint(1001, 10000)
        else:
            dst_ip = random.randint(1, 500)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 443
        else:
            dst_port = random.randint(1025, 65535)
    elif alert_type == 6:  #'domain_flux_alert'
        prob = random.randint(0, 100)
        if prob <= 70:
            dst_ip = random.randint(1, 500)
        else:
            dst_ip = random.randint(1001, 10000)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 53
        else:
            dst_port = random.randint(1025, 65535)
    elif alert_type == 7:  #'scan_alert'
        dst_ip = random.randint(1, 500)
        dst_port = random.randint(1, 1024)
        prob = random.randint(0, 100)
        if prob <= 90:
            scanned_host = dst_ip
        else:
            scanned_host = random.randint(1, 500)
    else:  #'tor_alert'
        dst_ip = random.randint(10001, 10100) # tor server list
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 443
        else:
            dst_port = random.randint(1025, 65535)

    prob = random.randint(0, 100)
    if prob <= 90:
        infected_host = src_ip
    else:
        infected_host = random.randint(1, 500)

    random_alerts.loc[i] = np.array([alert_type, timestamp, src_ip, src_port, dst_ip, dst_port, infected_host, scanned_host])

# 4 STAGE APT GENERATION -----------------------------------------------------------------------------------------------
# 4 STEPS IN APT LIFE CYCLE MUST BE DETECTED

APT_alerts_full = pd.DataFrame(np.zeros((400, 8)), columns=columns)

for i in range(0, 400, 4):
    timestamp = random.randint(1, 1296000)
    src_ip = random.randint(1, 500)  # in campus network
    src_port = random.randint(49152, 65535)
    scanned_host = 0
    A_prob = random.randint(0, 100)

    # ******* POINT OF ENTRY ALERT GENERATION STEP

    if A_prob <= 33:
        alert_type = 1
        prob = random.randint(0, 100)
        if A_prob <= 70:
            dst_ip = random.randint(1001, 10000)  # fuori organizzazione
        else:
            dst_ip = random.randint(1, 500)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 80
        else:
            dst_port = random.randint(1025, 65535)
    elif A_prob > 33 and A_prob <= 66:
        alert_type = 2
        prob = random.randint(0, 100)
        if prob <= 70:
            dst_ip = random.randint(1001, 10000)  # esterno all'organizzazione
        else:
            dst_ip = random.randint(1, 500)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 80
        else:
            dst_port = random.randint(1025, 65535)
    else:
        alert_type = 3
        prob = random.randint(0, 100)
        if prob <= 70:
            dst_ip = random.randint(1, 500)  # nell organizzazione
        else:
            dst_ip = random.randint(1001, 10000)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 53
        else:
            dst_port = random.randint(1025, 65535)

    prob = random.randint(0, 100)
    if prob <= 90:
        infected_host = src_ip
    else:
        infected_host = random.randint(1, 500)

    APT_alerts_full.loc[i] = np.array([alert_type, timestamp, src_ip, src_port, dst_ip, dst_port, infected_host, scanned_host])

    # ****** C2 ALERT GENERATION

    timestamp = random.randint(timestamp + 1, 1296000)
    src_port = random.randint(49152, 65535)
    scanned_host = 0
    B_prob = random.randint(0, 100)

    if B_prob <= 33:
        alert_type = 4
        prob = random.randint(0, 100)
        if prob <= 70:
            dst_ip = random.randint(1001, 1100)  # nella ip blacklist
        else:
            dst_ip = random.randint(1101, 10000)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 443
        else:
            dst_port = random.randint(1025, 65535)
    elif B_prob > 33 and B_prob <= 66:
        alert_type = 5
        prob = random.randint(0, 100)
        if prob <= 70:
            dst_ip = random.randint(1001, 10000)
        else:
            dst_ip = random.randint(1, 500)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 443
        else:
            dst_port = random.randint(1025, 65535)
    else:
        alert_type = 6
        prob = random.randint(0, 100)
        if prob <= 70:
            dst_ip = random.randint(1, 500)
        else:
            dst_ip = random.randint(1001, 10000)
        prob_port = random.randint(0, 100)
        if prob_port <= 90:
            dst_port = 53
        else:
            dst_port = random.randint(1025, 65535)

    prob = random.randint(0, 100)
    if prob <= 90:
        infected_host = src_ip
    else:
        infected_host = random.randint(1, 500)

    APT_alerts_full.loc[i + 1] = np.array([alert_type, timestamp, src_ip, src_port, dst_ip, dst_port, infected_host, scanned_host])

    # ******** DATA DISCOVERY ALERT GENERATION

    alert_type = 7
    timestamp = random.randint(timestamp + 1, 1296000)
    src_port = random.randint(49152, 65535)
    dst_ip = random.randint(1, 500)
    dst_port = random.randint(1, 1024)

    prob = random.randint(0, 100)
    if prob <= 90:
        scanned_host = dst_ip
    else:
        scanned_host = random.randint(1, 500)

    prob = random.randint(0, 100)
    if prob <= 90:
        infected_host = src_ip
    else:
        infected_host = random.randint(1, 500)

    APT_alerts_full.loc[i + 2] = np.array([alert_type, timestamp, src_ip, src_port, dst_ip, dst_port, infected_host, scanned_host])

    # ******** DATA EXFILTRATION ALERT GENERATION

    alert_type = 8
    timestamp = random.randint(timestamp + 1, 1296000)
    src_port = random.randint(49152, 65535)
    scanned_host = 0
    dst_ip = random.randint(10001, 10100)  # tor server list
    prob_port = random.randint(0, 100)
    if prob_port <= 90:
        dst_port = 443
    else:
        dst_port = random.randint(1025, 65535)

    prob = random.randint(0, 100)
    if prob <= 90:
        infected_host = src_ip
    else:
        infected_host = random.randint(1, 500)

    APT_alerts_full.loc[i + 3] = np.array([alert_type, timestamp, src_ip, src_port, dst_ip, dst_port, infected_host, scanned_host])


print("done")