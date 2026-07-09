import numpy as np


def load_dataset(subject: int = 1):
    from moabb.datasets import BNCI2014_004
    from moabb.paradigms import LeftRightImagery

    paradigm = LeftRightImagery()
    dataset = BNCI2014_004()
    X, y_str, meta = paradigm.get_data(dataset, subjects=[subject])

    label_map = {"left_hand": 0, "right_hand": 1}
    y = np.array([label_map[lbl] for lbl in y_str])

    is_test = meta["session"].str.endswith("test").to_numpy()

    X_train, y_train = X[~is_test], y[~is_test]
    X_test,  y_test  = X[is_test],  y[is_test]

    train_sessions = sorted(meta["session"][~is_test].unique())
    test_sessions  = sorted(meta["session"][is_test].unique())
    print(
        f"[data] total={len(y)} trials, "
        f"train={len(y_train)} ({','.join(train_sessions)}), "
        f"test={len(y_test)} ({','.join(test_sessions)})"
    )

    return (
        X_train.astype("float32"),
        y_train,
        X_test.astype("float32"),
        y_test,
    )
