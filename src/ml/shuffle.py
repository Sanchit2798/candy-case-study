from copy import deepcopy

def shuffle(X, Y, random_state = 123):
    XY = deepcopy(X)
    XY['Y'] = Y
    # print(XY)
    XY = XY.sample(frac = 1, random_state = random_state).reset_index(drop=True)
    # print(XY)
    X = XY.drop('Y', axis = 1)
    Y = XY['Y']
    return X,Y