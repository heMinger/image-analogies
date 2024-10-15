import numpy as np


def get_indices(i, j, N, h, w):
    # Calculate border
    border = int(np.floor(N / 2))

    # Calculate start and end indices for i (row)
    i_top = i - border
    pad_top = 0
    if i_top < 0:
        start_i = 0
        pad_top = - i_top
    else:
        start_i = i_top

    i_bot = i + border
    pad_bot = N - 1
    if i_bot >= h:
        pad_bot = N - (i_bot - h) - 2
        end_i = h - 1
    else:
        end_i = i_bot

    # Calculate start and end indices for j (column)
    j_left = j - border
    pad_left = 0
    if j_left < 0:
        start_j = 0
        pad_left = - j_left
    else:
        start_j = j_left

    j_right = j + border
    pad_right = N - 1
    if j_right >= w:
        end_j = w - 1
        pad_right = N - (j_right - w) - 2
    else:
        end_j = j_right

    flag = False

    return int(start_i), int(end_i), int(start_j), int(end_j), int(pad_top), int(pad_bot), int(pad_left), int(pad_right), flag