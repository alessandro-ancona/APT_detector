import matplotlib.pyplot as plt

time_3_1 = [114, 255, 455, 851]
time_3_2 = [118, 258, 458, 809]
time_3_3 = [122, 277, 476, 861]

CE_3_1 = [128.2626, 0.0144, 0.0127, 0.0107]
CE_3_2 = [110.8878, 0.0144, 0.0121, 0.0104]
CE_3_3 = [162.4192, 0.0139, 0.0100, 0.0097]

A_3_1 = [0.9684, 0.9960, 0.9968, 0.9971]
A_3_2 = [0.9692, 0.9967, 0.9969, 0.9971]
A_3_3 = [0.9711, 0.9963, 0.9970, 0.9971]

LSTM = [0, 64, 128, 256]
def_x = range(len(LSTM))

# TIME vs L
plt.subplots(figsize=(6, 3), dpi=300)
plt.plot(def_x, time_3_1, '-ob', def_x, time_3_2, '-or', def_x, time_3_3, '-oy', linewidth="0.8")
plt.ylim([0, 1000])
plt.xticks(def_x, LSTM)
plt.xlabel("L", fontsize="medium")
plt.ylabel("Time [ms]", fontsize="medium")
plt.legend(('Architectures 3:1', 'Architectures 3:2', 'Architectures 3:3'), fontsize="small", loc='lower right', shadow=True)
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
plt.savefig("Epoch_time.eps")
plt.show()

# Accuracy vs L
plt.subplots(figsize=(6, 3), dpi=300)
plt.plot(def_x, A_3_1, '-ob', def_x, A_3_2, '-or', def_x, A_3_3, '-oy', linewidth="0.8")
plt.ylim([0.9, 1])
plt.xticks(def_x, LSTM)
plt.xlabel("L", fontsize="medium")
plt.ylabel("Accuracy", fontsize="medium")
plt.legend(('Architectures 3:1', 'Architectures 3:2', 'Architectures 3:3'), fontsize="small", loc='lower right', shadow=True)
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
plt.savefig("Accuracy_VS_L.eps")
plt.show()

# LOSS vs L
plt.subplots(figsize=(6, 3), dpi=300)
plt.plot(def_x, CE_3_1, '-ob', def_x, CE_3_2, '-or', def_x, CE_3_3, '-oy', linewidth="0.8")
plt.ylim([0, 170])
plt.xticks(def_x, LSTM)
plt.xlabel("L", fontsize="medium")
plt.ylabel("Loss", fontsize="medium")
plt.legend(('Architectures 3:1', 'Architectures 3:2', 'Architectures 3:3'), fontsize="small", loc='upper right', shadow=True)
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
plt.savefig("Loss_VS_L.eps")
plt.show()