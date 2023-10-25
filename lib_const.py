# coding: utf-8

class Const():
    D_MIN = 250 # m
    D_MAX = 350 # m
    # BETA_MAX = 2.5 # deg
    BETA_MAX = 15 # deg
    PHI = 850 # m
    NU = 3. # 博士論文中の「n」
    Y_LIM = [-4000, 4000] # m
    X_LIM = [-4000, 4000] # m
    EPOCH = 200
    SEED = 11

    MIN_LINK_LENGTH = 5e-3

    LOCAL_ROAD_FACTOR = 1.5 # 地区道路を φ の何倍程度の長さまで許すか

    v = 1.5 # 近くに道路がある時に、地区道路を何倍まで延長してまでそこに着けるか


class DebugConst():
    PRINT_QN = False
    PRINT_Q0 = False