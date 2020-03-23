import numpy as np
from tqdm import tqdm

cards = [[10, 11], [20, 21], [30, 31], [40, 41]]


def test(num):
    positive = 0
    total = 0
    count = 0

    for _ in tqdm(range(num)):
        two_cards = np.random.choice([0, 1, 2, 3], 2)
        flip = [np.random.choice(cards[two_cards[0]]), np.random.choice(cards[two_cards[1]])]

        count += 1
        if flip[0] == 10 or flip[1] == 10:
            total += 1

            if flip[0] in (10, 20, 30, 40) and flip[1] in (10, 20, 30, 40):
                positive += 1

    return count, total, positive, total/count, positive/total, 15/64, 7/15


# # todo: move this into matardn
# def generate_matrix(out_height, out_width, scale):
#     height = t.linspace(0, (out_height - 1) / scale, out_height)
#     width = t.linspace(0, (out_width - 1) / scale, out_width)
#     r_1 = t.tensor([[i, j, 1 / scale] for i in height for j in width])
#     r_2 = t.tensor([[int(i / scale), int(j / scale), 0] for i in height for j in width])
#     return r_1 - r_2


if __name__ == '__main__':
    print(test(1000000))
