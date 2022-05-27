import numpy as np

np.random.seed(123)


class APTDetector:
    """
    Class to perform alert clustering and detection
    """

    def __init__(self, dataset, corr_window=1e5):
        self.corr_window = corr_window
        self.data_size = dataset.shape
        self.dataset = dataset

    def cluster(self, cluster_size=4):
        clusters = {}
        n = 0
        for i in range(self.data_size[0]):
            curr_alert = self.dataset[i, :]
            alert_type = curr_alert[0]
            timestamp = curr_alert[1]
            infected_host = curr_alert[6]
            if alert_type == 1 or alert_type == 2 or alert_type == 3:  # alert_1
                new_cluster = np.zeros((cluster_size, self.data_size[1]))
                new_cluster[0, :] = curr_alert
                clusters[n] = new_cluster
                n = n + 1
            elif alert_type == 4 or alert_type == 5 or alert_type == 6:  # alert_2
                for cluster in clusters.values():

                    if (cluster[1, :] - np.zeros(self.data_size[1])).any():  # se esiste già un alert_2
                        continue
                    else:
                        alert_1 = cluster[0, :]
                        if (alert_1 - np.zeros(self.data_size[1])).any():  # cioè se esiste in questo cluster un alert_1
                            if timestamp > alert_1[1] and infected_host == alert_1[
                                6]:  # e tutte le condizioni sono vere
                                cluster[1, :] = curr_alert  # crea alert_2
                                break
                            else:
                                if not ((cluster - clusters[n - 1]).any()):  # se è ultimo cluster allora crea uno nuovo
                                    new_cluster = np.zeros((cluster_size, self.data_size[1]))
                                    new_cluster[1, :] = curr_alert
                                    clusters[n] = new_cluster
                                    n = n + 1
                                    break  # il break deve essere quando crei il nuovo cluster
                                else:
                                    continue

                        elif not ((cluster - clusters[
                            n - 1]).any()):  # non esiste alert 1 in questo cluster, allora se ì l'ultimo cluster, aggiungi nuovo altrimenti vai avanti con il for
                            new_cluster = np.zeros((cluster_size, self.data_size[1]))
                            new_cluster[1, :] = curr_alert
                            clusters[n] = new_cluster
                            n = n + 1
                            break
                        else:
                            continue

            elif alert_type == 7:  # alert_3
                for cluster in clusters.values():
                    if (cluster[2, :] - np.zeros(
                            self.data_size[1])).any():  # se esiste già un alert_3 per questo cluster vai avanti
                        continue
                    else:
                        alert_1 = cluster[0, :]
                        alert_2 = cluster[1, :]
                        if (alert_1 - np.zeros(self.data_size[1])).any() and (
                                alert_2 - np.zeros(self.data_size[1])).any():  # se esiste alert 1 e 2
                            if timestamp > alert_2[1] and infected_host == alert_2[6]:
                                cluster[2, :] = curr_alert  # crea alert_3
                                break
                            else:
                                if not ((cluster - clusters[n - 1]).any()):  # se è ultimo cluster allora crea uno nuovo
                                    new_cluster = np.zeros((cluster_size, self.data_size[1]))
                                    new_cluster[2, :] = curr_alert
                                    clusters[n] = new_cluster
                                    n = n + 1
                                    break  # il break deve essere quando crei il nuovo cluster
                                else:
                                    continue
                        elif (alert_1 - np.zeros(self.data_size[1])).any():  # se esiste solo alert 1
                            if timestamp > alert_1[1] and infected_host == alert_1[6]:
                                cluster[2, :] = curr_alert  # crea alert_3
                                break
                            else:
                                if not ((cluster - clusters[n - 1]).any()):  # se è ultimo cluster allora crea uno nuovo
                                    new_cluster = np.zeros((cluster_size, self.data_size[1]))
                                    new_cluster[2, :] = curr_alert
                                    clusters[n] = new_cluster
                                    n = n + 1
                                    break  # il break deve essere quando crei il nuovo cluster
                                else:
                                    continue
                        elif (alert_2 - np.zeros(self.data_size[1])).any():  # se esiste solo alert 2
                            if timestamp > alert_2[1] and infected_host == alert_2[6]:
                                cluster[2, :] = curr_alert  # crea alert_3
                                break
                            else:
                                if not ((cluster - clusters[n - 1]).any()):  # se è ultimo cluster allora crea uno nuovo
                                    new_cluster = np.zeros((cluster_size, self.data_size[1]))
                                    new_cluster[2, :] = curr_alert
                                    clusters[n] = new_cluster
                                    n = n + 1
                                    break  # il break deve essere quando crei il nuovo cluster
                                else:
                                    continue
                        elif not ((cluster - clusters[
                            n - 1]).any()):  # se non esiste nessun alert 1 o 2 e se è ultimo cluster crea nuovo altrimenti go on
                            new_cluster = np.zeros(cluster_size, self.data_size[1])
                            new_cluster[2, :] = curr_alert
                            clusters[n] = new_cluster
                            n = n + 1
                            break
                        else:
                            break
            else:  # alert_4
                for cluster in clusters.values():
                    if (cluster[3, :] - np.zeros(
                            self.data_size[1])).any():  # se esiste già un alert_4 per questo cluster vai avanti
                        continue
                    else:
                        alert_1 = cluster[0, :]
                        alert_2 = cluster[1, :]
                        alert_3 = cluster[2, :]
                        if (alert_1 - np.zeros(self.data_size[1])).any() and (
                                alert_2 - np.zeros(self.data_size[1])).any() and (
                                alert_3 - np.zeros(self.data_size[1])).any():
                            if timestamp > alert_3[1] and infected_host == alert_3[6]:
                                cluster[3, :] = curr_alert  # crea alert_4
                                break
                        elif (alert_1 - np.zeros(self.data_size[1])).any() and (
                                alert_2 - np.zeros(self.data_size[1])).any():
                            if timestamp > alert_2[1] and infected_host == alert_2[6]:
                                cluster[3, :] = curr_alert  # crea alert_4
                                break
                        elif (alert_1 - np.zeros(self.data_size[1])).any() and (
                                alert_3 - np.zeros(self.data_size[1])).any():
                            if timestamp > alert_3[1] and infected_host == alert_3[6]:
                                cluster[3, :] = curr_alert  # crea alert_4
                                break
                        elif (alert_2 - np.zeros(self.data_size[1])).any() and (
                                alert_3 - np.zeros(self.data_size[1])).any():
                            if timestamp > alert_3[1] and infected_host == alert_3[6]:
                                cluster[3, :] = curr_alert  # crea alert_4
                                break
                        elif (alert_1 - np.zeros(self.data_size[1])).any():
                            if timestamp > alert_1[1] and infected_host == alert_1[6]:
                                cluster[3, :] = curr_alert  # crea alert_4
                                break
                        elif (alert_2 - np.zeros(self.data_size[1])).any():
                            if timestamp > alert_2[1] and infected_host == alert_2[6]:
                                cluster[3, :] = curr_alert  # crea alert_4
                                break
                        elif (alert_3 - np.zeros(self.data_size[1])).any():
                            if timestamp > alert_3[1] and infected_host == alert_3[6]:
                                cluster[3, :] = curr_alert  # crea alert_4
                                break
                        elif not ((cluster == clusters[n - 1]).any()):
                            return clusters
                        else:
                            break
        return clusters

    def score(self, clusters):
        data = np.zeros((1, 18))
        i = 0
        for cluster in clusters.values():  # at least alert_1
            alert_1 = cluster[0, :]
            alert_2 = cluster[1, :]
            alert_3 = cluster[2, :]
            alert_4 = cluster[3, :]
            if (alert_1 - np.zeros(self.data_size[1])).any():
                if (alert_2 - np.zeros(self.data_size[1])).any():
                    if (alert_3 - np.zeros(self.data_size[1])).any():
                        if (alert_4 - np.zeros(self.data_size[1])).any():  # è APT
                            new_row = np.concatenate((alert_1, alert_2))
                            new_row = np.append(new_row, np.array([3]))
                            new_row = np.append(new_row, np.array([1]))
                            data[i, :] = new_row
                            data = np.append(data, np.zeros((1, 18)), axis=0)
                            i = i + 1
                        else:
                            new_row = np.concatenate((alert_1, alert_2))
                            new_row = np.append(new_row, np.array([2]))
                            new_row = np.append(new_row, np.array([0]))
                            data[i, :] = new_row
                            data = np.append(data, np.zeros((1, 18)), axis=0)
                            i = i + 1
                    else:
                        new_row = np.concatenate((alert_1, alert_2))
                        new_row = np.append(new_row, np.array([1]))
                        new_row = np.append(new_row, np.array([0]))
                        data[i, :] = new_row
                        data = np.append(data, np.zeros((1, 18)), axis=0)
                        i = i + 1
                else:
                    continue
            else:
                continue
        data = np.delete(data, len(data)-1, 0)
        return data
