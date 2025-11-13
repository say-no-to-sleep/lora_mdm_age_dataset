import numpy as np
import os


def main() -> None:
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "Comp_v6_KLD01")
    std = np.load(os.path.join(data_dir, "Std.npy"))
    mean = np.load(os.path.join(data_dir, "Mean.npy"))

    np.set_printoptions(linewidth=120)

    std_zero = np.argwhere(std == 0)
    mean_zero = np.argwhere(mean == 0)

    if std_zero.size == 0:
        print("Std: no zero entries found.")
    else:
        print("Std zero-value indices (np.argwhere order):")
        print(std_zero)

    if mean_zero.size == 0:
        print("Mean: no zero entries found.")
    else:
        print("Mean zero-value indices (np.argwhere order):")
        print(mean_zero)


if __name__ == "__main__":
    main()
