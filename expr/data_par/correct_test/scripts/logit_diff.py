import numpy as np


def get_logit_diff():
    logit_base = np.load("./logit_base.npy")
    logit_ooo = np.load("./logit_ooo.npy")

    print(f"logit_base : {logit_base}")
    print(f"logit_ooo : {logit_ooo}")

    logit_diff = logit_base - logit_ooo
    avg_logit_diff = np.mean(logit_diff)
    
    print(f"logit_diff : {logit_diff}")


if __name__ == "__main__":
    get_logit_diff()

