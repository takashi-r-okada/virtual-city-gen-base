# coding: utf-8

import numpy as np 
from pathlib import Path 
import shutil
import cv2
import matplotlib.pyplot as plt
from enum import Enum
import math
import random
from lib_const import Const, DebugConst

np.random.seed(Const.SEED)
random.seed(Const.SEED)

#TODO 道路近傍場の実装と、実際に近くに道路があると判明したときの収斂
#TODO ぶち当たった時にそこを新たに交差点とし、もとからある道路の link を 2 分割する


areas = []


class LinkType(Enum):
    """
    - F: 幹線道路 (先端)．
    - A: 幹線道路 (先端以外)．
    - B 系: 地区道路．先端状態で残さずにどこか幹線道路まで繋がるような状態に常にする．
      + B: 地区道路 (先端以外)．これ以上変化しない．
      + B0: 直進系地区道路
      + B1: 時計回りで作られた地区道路 (最早路地)
      + B2: 反時計回りで作られた地区道路 (最早路地)
    """
    F = 10
    A = 20
    B = 30
    B0 = 31
    B1 = 32
    B2 = 33

def newNAND(t_a, t_b):
    """
    初めて見るタイプの新 NAND
    """
    return ((t_a & t_b) == 0)

def newAND(t_a, t_b):
    """
    初めて見るタイプの新 NND
    """
    return ((t_a & t_b) != 0)

# class Point():
#     # 点。曲がる点だったり、交差点だったり、開放端だったり、勾配の定義されている場所だったりする。

#     def __init__(self, y: float, x: float):
#         self.POS = np.array([y, x]) # m


def direc2arg(direc: np.ndarray):
    '''ベクトルから、arg (x 軸正の方向とのなす角、0 <= arg < 360, deg) を求める'''
    return np.rad2deg(np.arctan2(direc[0], direc[1])) % 360.

def arg2Direc(arg: float):
    '''arg (deg) から単位ベクトルを求める'''
    assert 0 <= arg < 360.

    ret = np.array([np.tan(np.deg2rad(arg)), 1.0])

    if 90. < arg < 270.:
        ret = -ret

    ret = ret / (np.sum(ret ** 2.) ** 0.5)
    return ret
        

class Link():
    """
    Link とは有向線分であると共に、END_POS そのものである。
    """
    def __init__(
        self, 
        linkType: LinkType, 
        startPos: np.ndarray, 
        # startDirec: np.ndarray, 
        theta_0: float, # deg
        t: int=0,
    ):
        
        self.LINK_TYPE = linkType
        self.LENGTH = np.random.uniform(low=Const.D_MIN, high=Const.D_MAX)

        self.THETA = (theta_0 + np.random.uniform(low=-Const.BETA_MAX, high=Const.BETA_MAX)) % 360.
        # self.THETA = (theta_0 + np.random.normal(loc=0, scale=Const.BETA_MAX/2.)) % 360.
         
        self.DIREC = arg2Direc(self.THETA) # 単位ベクトル

        self.START_POS = startPos
        self.END_POS = startPos + self.LENGTH * self.DIREC

        # print(self.START_POS, self.END_POS)

        self.t = t # これは他と違って変数。しかし単調増加であって減ることはない。

        assert self.LENGTH > 1e-3
        assert np.sum((self.END_POS - self.START_POS) ** 2)**0.5 > 1e-3

    def __repr__(self):
        linkTypeStr = ""
        if self.LINK_TYPE == LinkType.F:
            linkTypeStr = "F"
        elif self.LINK_TYPE == LinkType.A:
            linkTypeStr = "A"
        elif self.LINK_TYPE == LinkType.B:
            linkTypeStr = "B"
        elif self.LINK_TYPE == LinkType.B0:
            linkTypeStr = "B0"
        elif self.LINK_TYPE == LinkType.B1:
            linkTypeStr = "B1"
        elif self.LINK_TYPE == LinkType.B2:
            linkTypeStr = "B2"
        return "<%s:(%d,%d)->(%d,%d)>" % (
            linkTypeStr,
            self.START_POS[1],
            self.START_POS[0],
            self.END_POS[1],
            self.END_POS[0],
        )

    def getT(self):
        """
        END_POS (屈折点・交差点) の発展場 T の値。
        T は C_n を決定する引数となる。(また、翻って G_i を決定する引数ともなる)

        cf. Area.getT
        """

        T_candidates = [0.01,] 
        # T_candidates = [0.05, ] # debug 用
        for a in areas:
            # print(a.G * (1. - ((self.END_POS - a.GC)/Const.D_MAX) ** Const.NU))
            T_candidates.append(
                a.G * (1. - (
                    (np.sum((self.END_POS - a.GC) ** 2.) ** .5)
                    /Const.D_MAX
                ) ** Const.NU)
            )

        # print(T_candidates)
        T = max(T_candidates)
        T = np.clip(T, 0., 1.)
        return T

    def getN(self):
        '''
        Link の END_POS 点の幹線道路の場
        '''
        # ここで、self.END_POS を使って true か false かを求める
        minLength = 100000
        flattenRoadNet = flatten(roadNet, Link)
        nearestLink = None

        n = False

        for tgtLink in flattenRoadNet:

            # # 直線の式 ax + by + c = 0 の a, b, c を求める
            # _a = tgtLink.START_POS[0] - tgtLink.END_POS[0]
            # _b = -tgtLink.START_POS[1] + tgtLink.END_POS[1]
            # _c = -tgtLink.START_POS[1] * tgtLink.END_POS[0] - tgtLink.START_POS[0] * tgtLink.END_POS[1] + 2 * tgtLink.END_POS[0] * tgtLink.END_POS[1]

            # _distanceToTgtLink = abs(_a*self.END_POS[1]+_b*self.END_POS[0]+_c) / ((_a**2. + _b**2.) ** 0.5)

            # if _distanceToTgtLink > Const.D_MIN * 0.5:
            #     continue

            # if minDistanceToNearestLink >

            if not(tgtLink.LINK_TYPE in [LinkType.F, LinkType.A]):
                continue

            _length, _ = self.getLengthEncounteringTo(tgtLink)
            if _length is None:
                continue
            elif _length > self.LENGTH * 3.:
                continue
            elif _length < self.LENGTH * 1.:
                continue

            n = True

            if _length < minLength:
                minLength = _length
                nearestLink = tgtLink

            del _length

        return n, minLength, nearestLink

            


    def getN_C(self):
        '''
        Link の END_POS 点の幹線道路同士の交差点の場。半径 φ に幹線道路同士の交差点があれば True, そうでなければ False
        '''
        flattenRoadNet = flatten(roadNet, Link)
        for tgtLink in flattenRoadNet:
            if tgtLink.LINK_TYPE in [LinkType.A, LinkType.F]:
                if newAND(tgtLink.t, 12):
                    if np.sum((tgtLink.END_POS - self.END_POS) ** 2.) ** 0.5 < Const.PHI:
                        return True
        return False




    def getStepped(self):
        '''
        自身を含めて破壊的に変更することで 1 step 踏む
        '''

        choicesList = []

        if self.LINK_TYPE == LinkType.F:
            _n, _, _= self.getN()
            if _n:
                choicesList.append(Q[2])
                choicesList.append(Q[3])
            else:
                choicesList.append(Q[1])

        elif self.LINK_TYPE == LinkType.A:

            if not(self.getN_C()):

                if newNAND(self.t, 12):
                    choicesList.append(Q[4])
                if newNAND(self.t, 4):
                    choicesList.append(Q[5])
                if newNAND(self.t, 8):
                    choicesList.append(Q[6])

            # if newNAND(self.t, 1):
            if newNAND(self.t, 5): # 博士論文とは違うけど多分こっちでは？
                choicesList.append(Q[7])
            # if newNAND(self.t, 2):
            if newNAND(self.t, 10): # 博士論文とは違うけど多分こっちでは？
                choicesList.append(Q[8])
            # if newNAND(self.t, 1):
            if newNAND(self.t, 5): # 博士論文とは違うけど多分こっちでは？
                choicesList.append(Q[9])
            # if newNAND(self.t, 2):
            if newNAND(self.t, 10): # 博士論文とは違うけど多分こっちでは？
                choicesList.append(Q[10])

            # print(choicesList)

        elif self.LINK_TYPE == LinkType.B0:

            # if newNAND(self.t, 3):
            if newNAND(self.t, 15): # 博士論文とは違うけど多分こっちでは？
                choicesList.append(Q[11])
            # if newNAND(self.t, 1):
            if newNAND(self.t, 5): # 博士論文とは違うけど多分こっちでは？
                choicesList.append(Q[12])
            # if newNAND(self.t, 2):
            if newNAND(self.t, 10): # 博士論文とは違うけど多分こっちでは？
                choicesList.append(Q[13])

        elif self.LINK_TYPE == LinkType.B1:

            # if newNAND(self.t, 2):
            if newNAND(self.t, 10): # 博士論文とは違うけど多分こっちでは？
                choicesList.append(Q[14])

        elif self.LINK_TYPE == LinkType.B2:

            # if newNAND(self.t, 1):
            if newNAND(self.t, 5): # 博士論文とは違うけど多分こっちでは？
                choicesList.append(Q[15])
            
        else:
            assert False

        T = self.getT()

        weightsList = list(map(lambda ch: ch['alpha'] * T + ch['k'], choicesList))
        choicesList.append(None)
        weightsList.append(1. - np.sum(weightsList))

        # if self.LINK_TYPE == LinkType.A:
        #     print(choicesList, weightsList)
        # print(self, choicesList, weightsList)

        # 置き換え関数 replace の決定
        choosedDict = random.choices(choicesList, k=1, weights=weightsList)[0]
        if choosedDict is None:
            if DebugConst.PRINT_Q0:
                print('q0', [self, ])
            return [self, ]
        
        replace = choosedDict['replace']
        # print(replace)
        
        del choicesList
        del weightsList

        return replace(self)
    
    def getLengthEncounteringTo(self, tgtLink):
        """
        tgtLink にぶちあたる場合の length > 0 を求める
        tgtLink は線分とする。なので、ぶちあたらない場合は None を返す。

        [戻り値]
        - l (float): ぶち当たる場合の self.LENGTH
        - p * tgtLink.LENGTH (float): ぶち当たる場合の tgtLink の始点から交差点までの長さ
        """

        if self is tgtLink:
            return [None, None]
            

        A = np.array([
            [self.DIREC[0], tgtLink.START_POS[0] - tgtLink.END_POS[0]],
            [self.DIREC[1], tgtLink.START_POS[1] - tgtLink.END_POS[1]]
        ])
        b = np.array([
            tgtLink.START_POS[0] - self.START_POS[0],
            tgtLink.START_POS[1] - self.START_POS[1]
        ])
        # try:
        l, p = np.linalg.solve(A, b)
        # except:
        #     print(self, tgtLink)

        if not((l > 0) and (0 <= p <= 1)):
            return [None, None]
        
        del A
        del b
        
        return [l, p * tgtLink.LENGTH]    

def flatten(lst: list, elemClass):
    if len(lst) == 0:
        return []
    else:
        if type(lst[0]) is elemClass:
            return [lst[0],] + flatten(lst[1:],elemClass )
        else:
            return flatten(lst[0], elemClass) + flatten(lst[1:], elemClass)

# --------------------------------------------------
# 各置換関数の定義
# --------------------------------------------------

def q1(l):
    if DebugConst.PRINT_QN:
        print('q1')
    l.LINK_TYPE = LinkType.A
    sucL1 = Link(linkType=LinkType.F, startPos=l.END_POS, theta_0=l.THETA, t=0)
    return [l, sucL1]

def q2(l):
    if DebugConst.PRINT_QN:
        print('q2')
    l.LINK_TYPE = LinkType.A
    _, _, _nearestLink= l.getN()

    # sucL1 の伸縮をする。
    isConnected = False
    while not(isConnected):
        sucL1 = Link(linkType=LinkType.A, startPos=l.END_POS, theta_0=l.THETA, t=0)
        sucL1.LENGTH, _ = sucL1.getLengthEncounteringTo(_nearestLink)
        if (sucL1.LENGTH is not None):
            isConnected = True
        else:
            del sucL1
    del isConnected

    sucL1.END_POS = sucL1.START_POS + sucL1.DIREC * sucL1.LENGTH
    sucL1.t = 12

    return [l, sucL1]

def q3(l):
    if DebugConst.PRINT_QN:
        print('q3')
    l.LINK_TYPE = LinkType.A
    _, _, _nearestLink= l.getN()

    # sucL1 の伸縮をする。
    isConnected = False
    while not(isConnected):
        sucL1 = Link(linkType=LinkType.A, startPos=l.END_POS, theta_0=l.THETA, t=0)
        sucL1.LENGTH, _ = sucL1.getLengthEncounteringTo(_nearestLink)
        if (sucL1.LENGTH is not None):
            isConnected = True
        else:
            del sucL1
    del isConnected

    sucL1.END_POS = sucL1.START_POS + sucL1.DIREC * sucL1.LENGTH
    sucL1.t = 12

    return [l, sucL1]

def q4(l):
    if DebugConst.PRINT_QN:
        print('q4')
    l.t = 12

    sucL1 = Link(linkType=LinkType.F, startPos=l.END_POS, theta_0=l.THETA + 270, t=0)
    sucL2 = Link(linkType=LinkType.F, startPos=l.END_POS, theta_0=l.THETA + 90, t=0)

    return [l, [sucL1, ], [sucL2, ]]

def q5(l):
    if DebugConst.PRINT_QN:
        print('q5')
    l.t = (l.t & 14) | 4

    sucL1 = Link(linkType=LinkType.F, startPos=l.END_POS, theta_0=l.THETA + 270, t=0)

    return [l, [sucL1, ], ]

def q6(l):
    if DebugConst.PRINT_QN:
        print('q6')
    l.t = (l.t & 7) | 8

    sucL1 = Link(linkType=LinkType.F, startPos=l.END_POS, theta_0=l.THETA + 90, t=0)

    return [l, [sucL1, ], ]

def q7(l):
    if DebugConst.PRINT_QN:
        print('q7')
    l.t = (l.t) | 1

    b0_chain = []
    isEncountered = False
    isTerminated = False

    _sucL = Link(linkType=LinkType.B0, startPos=l.END_POS, theta_0=l.THETA + 270, t=0)
    b0_chain.append(_sucL)

    # - - - - - 以下 q7, q8 共通部分 - - - - - - - - - 
    flattenRoadNet = flatten(roadNet, Link)
    
    minLength = 10000
    tgtLinkType = None
    for tgtLink in flattenRoadNet:
        _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
        if _length is None:
            continue
        elif _length > b0_chain[-1].LENGTH * Const.v:
            continue
        elif _length < Const.MIN_LINK_LENGTH:
            continue

        isEncountered = True

        if _length < minLength:
            minLength = _length
            tgtLinkType = tgtLink.LINK_TYPE
        del _length

    while not(isEncountered or isTerminated):

        _sucL = Link(linkType=LinkType.B0, startPos=b0_chain[-1].END_POS, theta_0=b0_chain[-1].THETA, t=0)
        b0_chain.append(_sucL)

        minLength = 10000
        tgtLinkType = None
        for tgtLink in flattenRoadNet:
            _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
            if _length is None:
                continue
            elif _length > b0_chain[-1].LENGTH * Const.v:
                continue
            elif _length < Const.MIN_LINK_LENGTH:
                continue
            
            isEncountered = True

            if _length < minLength:
                minLength = _length
                tgtLinkType = tgtLink.LINK_TYPE
            del _length

        if not(isEncountered):
            if not(
                Const.Y_LIM[0] <= b0_chain[-1].END_POS[0] <= Const.Y_LIM[1] 
                and Const.X_LIM[0] <= b0_chain[-1].END_POS[1] <= Const.X_LIM[1]
            ):
                isTerminated = True

            if len(b0_chain) >= round(Const.PHI*Const.LOCAL_ROAD_FACTOR / ((Const.D_MIN + Const.D_MAX)/2)):
                isTerminated = True
    # - - - - - 以上 q7, q8 共通部分 - - - - - - - - - 

    if isEncountered:
        b0_chain[-1].LENGTH = minLength
        b0_chain[-1].END_POS = b0_chain[-1].START_POS + minLength * b0_chain[-1].DIREC
        b0_chain[-1].t = 12 if tgtLinkType in [LinkType.A, LinkType.F] else 3
        return [l, b0_chain]
    if isTerminated:
        return [l, b0_chain]


    

def q8(l):
    if DebugConst.PRINT_QN:
        print('q8')
    l.t = (l.t) | 2

    b0_chain = []
    isEncountered = False
    isTerminated = False

    _sucL = Link(linkType=LinkType.B0, startPos=l.END_POS, theta_0=l.THETA + 90, t=0)
    b0_chain.append(_sucL)

    # - - - - - 以下 q7, q8 共通部分 - - - - - - - - - 
    flattenRoadNet = flatten(roadNet, Link)
    
    minLength = 10000
    tgtLinkType = None
    for tgtLink in flattenRoadNet:
        _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
        if _length is None:
            continue
        elif _length > b0_chain[-1].LENGTH * Const.v:
            continue
        elif _length < Const.MIN_LINK_LENGTH:
            continue

        isEncountered = True

        if _length < minLength:
            minLength = _length
            tgtLinkType = tgtLink.LINK_TYPE
        del _length

    while not(isEncountered or isTerminated):

        _sucL = Link(linkType=LinkType.B0, startPos=b0_chain[-1].END_POS, theta_0=b0_chain[-1].THETA, t=0)
        b0_chain.append(_sucL)

        minLength = 10000
        tgtLinkType = None
        for tgtLink in flattenRoadNet:
            _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
            if _length is None:
                continue
            elif _length > b0_chain[-1].LENGTH * Const.v:
                continue
            elif _length < Const.MIN_LINK_LENGTH:
                continue
            
            isEncountered = True

            if _length < minLength:
                minLength = _length
                tgtLinkType = tgtLink.LINK_TYPE
            del _length

        if not(isEncountered):
            if not(
                Const.Y_LIM[0] <= b0_chain[-1].END_POS[0] <= Const.Y_LIM[1] 
                and Const.X_LIM[0] <= b0_chain[-1].END_POS[1] <= Const.X_LIM[1]
            ):
                isTerminated = True

            if len(b0_chain) >= round(Const.PHI*Const.LOCAL_ROAD_FACTOR / ((Const.D_MIN + Const.D_MAX)/2)):
                isTerminated = True
    # - - - - - 以上 q7, q8 共通部分 - - - - - - - - - 

    if isEncountered:
        b0_chain[-1].LENGTH = minLength
        b0_chain[-1].END_POS = b0_chain[-1].START_POS + minLength * b0_chain[-1].DIREC
        b0_chain[-1].t = 12 if tgtLinkType in [LinkType.A, LinkType.F] else 3
        return [l, b0_chain]
    if isTerminated:
        return [l, b0_chain]

def q9(l):
    if DebugConst.PRINT_QN:
        print('q9')

    l.t = (l.t) | 1

    b0_chain = []
    isEncountered = False
    isTerminated = False

    _sucL = Link(linkType=LinkType.B1, startPos=l.END_POS, theta_0=l.THETA + 270, t=0)
    b0_chain.append(_sucL)


    flattenRoadNet = flatten(roadNet, Link)
    
    minLength = 10000
    tgtLinkType = None
    for tgtLink in flattenRoadNet:
        _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
        if _length is None:
            continue
        elif _length > b0_chain[-1].LENGTH * Const.v:
            continue
        elif _length < Const.MIN_LINK_LENGTH:
            continue

        isEncountered = True

        if _length < minLength:
            minLength = _length
            tgtLinkType = tgtLink.LINK_TYPE
        del _length

    while not(isEncountered or isTerminated):

        b0_chain[-1].t = (b0_chain[-1].t) | 1

        _sucL = Link(linkType=LinkType.B1, startPos=b0_chain[-1].END_POS, theta_0=b0_chain[-1].THETA + 270., t=0)
        b0_chain.append(_sucL)

        minLength = 10000
        tgtLinkType = None
        for tgtLink in flattenRoadNet:
            _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
            if _length is None:
                continue
            elif _length > b0_chain[-1].LENGTH * Const.v:
                continue
            elif _length < Const.MIN_LINK_LENGTH:
                continue
            
            isEncountered = True

            if _length < minLength:
                minLength = _length
                tgtLinkType = tgtLink.LINK_TYPE
            del _length

        if not(isEncountered):
            # if not(
            #     Const.Y_LIM[0] <= b0_chain[-1].END_POS[0] <= Const.Y_LIM[1] 
            #     and Const.X_LIM[0] <= b0_chain[-1].END_POS[1] <= Const.X_LIM[1]
            # ):
            #     isTerminated = True

            if len(b0_chain) >= 6:
                isTerminated = True

    if isEncountered:
        b0_chain[-1].LENGTH = minLength
        b0_chain[-1].END_POS = b0_chain[-1].START_POS + minLength * b0_chain[-1].DIREC
        b0_chain[-1].t = 12 if tgtLinkType in [LinkType.A, LinkType.F] else 3
        return [l, b0_chain]
    if isTerminated:
        return [l, b0_chain]

def q10(l):
    if DebugConst.PRINT_QN:
        print('q10')
    l.t = (l.t) | 2

    b0_chain = []
    isEncountered = False
    isTerminated = False

    _sucL = Link(linkType=LinkType.B2, startPos=l.END_POS, theta_0=l.THETA + 90, t=0)
    b0_chain.append(_sucL)


    flattenRoadNet = flatten(roadNet, Link)
    
    minLength = 10000
    tgtLinkType = None
    for tgtLink in flattenRoadNet:
        _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
        if _length is None:
            continue
        elif _length > b0_chain[-1].LENGTH * Const.v:
            continue
        elif _length < Const.MIN_LINK_LENGTH:
            continue

        isEncountered = True

        if _length < minLength:
            minLength = _length
            tgtLinkType = tgtLink.LINK_TYPE
        del _length

    while not(isEncountered or isTerminated):

        b0_chain[-1].t = (b0_chain[-1].t) | 2

        _sucL = Link(linkType=LinkType.B2, startPos=b0_chain[-1].END_POS, theta_0=b0_chain[-1].THETA + 90., t=0)
        b0_chain.append(_sucL)

        minLength = 10000
        tgtLinkType = None
        for tgtLink in flattenRoadNet:
            _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
            if _length is None:
                continue
            elif _length > b0_chain[-1].LENGTH * Const.v:
                continue
            elif _length < Const.MIN_LINK_LENGTH:
                continue
            
            isEncountered = True

            if _length < minLength:
                minLength = _length
                tgtLinkType = tgtLink.LINK_TYPE
            del _length

        if not(isEncountered):
            # if not(
            #     Const.Y_LIM[0] <= b0_chain[-1].END_POS[0] <= Const.Y_LIM[1] 
            #     and Const.X_LIM[0] <= b0_chain[-1].END_POS[1] <= Const.X_LIM[1]
            # ):
            #     isTerminated = True

            if len(b0_chain) >= 6:
                isTerminated = True

    if isEncountered:
        b0_chain[-1].LENGTH = minLength
        b0_chain[-1].END_POS = b0_chain[-1].START_POS + minLength * b0_chain[-1].DIREC
        b0_chain[-1].t = 12 if tgtLinkType in [LinkType.A, LinkType.F] else 3
        return [l, b0_chain]
    if isTerminated:
        return [l, b0_chain]


def q11(l):
    if DebugConst.PRINT_QN:
        print('q11')

    l.t = (l.t) | 3

    b0_chain = []
    isEncountered = False
    isTerminated = False

    _sucL = Link(linkType=LinkType.B0, startPos=l.END_POS, theta_0=l.THETA + 270, t=0)
    b0_chain.append(_sucL)

    # - - - - - 以下 q7, q8 共通部分 - - - - - - - - - 
    flattenRoadNet = flatten(roadNet, Link)
    
    minLength = 10000
    tgtLinkType = None
    for tgtLink in flattenRoadNet:
        _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
        if _length is None:
            continue
        elif _length > b0_chain[-1].LENGTH * Const.v:
            continue
        elif _length < Const.MIN_LINK_LENGTH:
            continue

        isEncountered = True

        if _length < minLength:
            minLength = _length
            tgtLinkType = tgtLink.LINK_TYPE
        del _length

    while not(isEncountered or isTerminated):

        _sucL = Link(linkType=LinkType.B0, startPos=b0_chain[-1].END_POS, theta_0=b0_chain[-1].THETA, t=0)
        b0_chain.append(_sucL)

        minLength = 10000
        tgtLinkType = None
        for tgtLink in flattenRoadNet:
            _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
            if _length is None:
                continue
            elif _length > b0_chain[-1].LENGTH * Const.v:
                continue
            elif _length < Const.MIN_LINK_LENGTH:
                continue
            
            isEncountered = True

            if _length < minLength:
                minLength = _length
                tgtLinkType = tgtLink.LINK_TYPE
            del _length

        if not(isEncountered):
            if not(
                Const.Y_LIM[0] <= b0_chain[-1].END_POS[0] <= Const.Y_LIM[1] 
                and Const.X_LIM[0] <= b0_chain[-1].END_POS[1] <= Const.X_LIM[1]
            ):
                isTerminated = True

            if len(b0_chain) >= round(Const.PHI*Const.LOCAL_ROAD_FACTOR / ((Const.D_MIN + Const.D_MAX)/2)):
                isTerminated = True
    # - - - - - 以上 q7, q8 共通部分 - - - - - - - - - 

    if isEncountered:
        b0_chain[-1].LENGTH = minLength
        b0_chain[-1].END_POS = b0_chain[-1].START_POS + minLength * b0_chain[-1].DIREC
        b0_chain[-1].t = 12 if tgtLinkType in [LinkType.A, LinkType.F] else 3

    # return [l, b0_chain]

    b0_chain_right = b0_chain

    b0_chain = []
    isEncountered = False
    isTerminated = False

    _sucL = Link(linkType=LinkType.B0, startPos=l.END_POS, theta_0=l.THETA + 90, t=0)
    b0_chain.append(_sucL)

    # - - - - - 以下 q7, q8 共通部分 - - - - - - - - - 
    flattenRoadNet = flatten(roadNet, Link)
    
    minLength = 10000
    tgtLinkType = None
    for tgtLink in flattenRoadNet:
        _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
        if _length is None:
            continue
        elif _length > b0_chain[-1].LENGTH * Const.v:
            continue
        elif _length < Const.MIN_LINK_LENGTH:
            continue

        isEncountered = True

        if _length < minLength:
            minLength = _length
            tgtLinkType = tgtLink.LINK_TYPE
        del _length

    while not(isEncountered or isTerminated):

        _sucL = Link(linkType=LinkType.B0, startPos=b0_chain[-1].END_POS, theta_0=b0_chain[-1].THETA, t=0)
        b0_chain.append(_sucL)

        minLength = 10000
        tgtLinkType = None
        for tgtLink in flattenRoadNet:
            _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
            if _length is None:
                continue
            elif _length > b0_chain[-1].LENGTH * Const.v:
                continue
            elif _length < Const.MIN_LINK_LENGTH:
                continue
            
            isEncountered = True

            if _length < minLength:
                minLength = _length
                tgtLinkType = tgtLink.LINK_TYPE
            del _length

        if not(isEncountered):
            if not(
                Const.Y_LIM[0] <= b0_chain[-1].END_POS[0] <= Const.Y_LIM[1] 
                and Const.X_LIM[0] <= b0_chain[-1].END_POS[1] <= Const.X_LIM[1]
            ):
                isTerminated = True

            if len(b0_chain) >= round(Const.PHI*Const.LOCAL_ROAD_FACTOR / ((Const.D_MIN + Const.D_MAX)/2)):
                isTerminated = True
    # - - - - - 以上 q7, q8 共通部分 - - - - - - - - - 

    if isEncountered:
        b0_chain[-1].LENGTH = minLength
        b0_chain[-1].END_POS = b0_chain[-1].START_POS + minLength * b0_chain[-1].DIREC
        b0_chain[-1].t = 12 if tgtLinkType in [LinkType.A, LinkType.F] else 3

    b0_chain_left = b0_chain

    return [l, b0_chain_right, b0_chain_left, ]

def q12(l):
    if DebugConst.PRINT_QN:
        print('q12')

    l.t = (l.t) | 1

    b0_chain = []
    isEncountered = False
    isTerminated = False

    _sucL = Link(linkType=LinkType.B0, startPos=l.END_POS, theta_0=l.THETA + 270, t=0)
    b0_chain.append(_sucL)

    # - - - - - 以下 q7, q8 共通部分 - - - - - - - - - 
    flattenRoadNet = flatten(roadNet, Link)
    
    minLength = 10000
    tgtLinkType = None
    for tgtLink in flattenRoadNet:
        _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
        if _length is None:
            continue
        elif _length > b0_chain[-1].LENGTH * Const.v:
            continue
        elif _length < Const.MIN_LINK_LENGTH:
            continue

        isEncountered = True

        if _length < minLength:
            minLength = _length
            tgtLinkType = tgtLink.LINK_TYPE
        del _length

    while not(isEncountered or isTerminated):

        _sucL = Link(linkType=LinkType.B0, startPos=b0_chain[-1].END_POS, theta_0=b0_chain[-1].THETA, t=0)
        b0_chain.append(_sucL)

        minLength = 10000
        tgtLinkType = None
        for tgtLink in flattenRoadNet:
            _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
            if _length is None:
                continue
            elif _length > b0_chain[-1].LENGTH * Const.v:
                continue
            elif _length < Const.MIN_LINK_LENGTH:
                continue
            
            isEncountered = True

            if _length < minLength:
                minLength = _length
                tgtLinkType = tgtLink.LINK_TYPE
            del _length

        if not(isEncountered):
            if not(
                Const.Y_LIM[0] <= b0_chain[-1].END_POS[0] <= Const.Y_LIM[1] 
                and Const.X_LIM[0] <= b0_chain[-1].END_POS[1] <= Const.X_LIM[1]
            ):
                isTerminated = True

            if len(b0_chain) >= round(Const.PHI*Const.LOCAL_ROAD_FACTOR / ((Const.D_MIN + Const.D_MAX)/2)):
                isTerminated = True
    # - - - - - 以上 q7, q8 共通部分 - - - - - - - - - 

    if isEncountered:
        b0_chain[-1].LENGTH = minLength
        b0_chain[-1].END_POS = b0_chain[-1].START_POS + minLength * b0_chain[-1].DIREC
        b0_chain[-1].t = 12 if tgtLinkType in [LinkType.A, LinkType.F] else 3

    return [l, b0_chain]

def q13(l):
    if DebugConst.PRINT_QN:
        print('q13')

    l.t = (l.t) | 2

    b0_chain = []
    isEncountered = False
    isTerminated = False

    _sucL = Link(linkType=LinkType.B0, startPos=l.END_POS, theta_0=l.THETA + 90, t=0)
    b0_chain.append(_sucL)

    # - - - - - 以下 q7, q8 共通部分 - - - - - - - - - 
    flattenRoadNet = flatten(roadNet, Link)
    
    minLength = 10000
    tgtLinkType = None
    for tgtLink in flattenRoadNet:
        _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
        if _length is None:
            continue
        elif _length > b0_chain[-1].LENGTH * Const.v:
            continue
        elif _length < Const.MIN_LINK_LENGTH:
            continue

        isEncountered = True

        if _length < minLength:
            minLength = _length
            tgtLinkType = tgtLink.LINK_TYPE
        del _length

    while not(isEncountered or isTerminated):

        _sucL = Link(linkType=LinkType.B0, startPos=b0_chain[-1].END_POS, theta_0=b0_chain[-1].THETA, t=0)
        b0_chain.append(_sucL)

        minLength = 10000
        tgtLinkType = None
        for tgtLink in flattenRoadNet:
            _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
            if _length is None:
                continue
            elif _length > b0_chain[-1].LENGTH * Const.v:
                continue
            elif _length < Const.MIN_LINK_LENGTH:
                continue
            
            isEncountered = True

            if _length < minLength:
                minLength = _length
                tgtLinkType = tgtLink.LINK_TYPE
            del _length

        if not(isEncountered):
            if not(
                Const.Y_LIM[0] <= b0_chain[-1].END_POS[0] <= Const.Y_LIM[1] 
                and Const.X_LIM[0] <= b0_chain[-1].END_POS[1] <= Const.X_LIM[1]
            ):
                isTerminated = True

            if len(b0_chain) >= round(Const.PHI*Const.LOCAL_ROAD_FACTOR / ((Const.D_MIN + Const.D_MAX)/2)):
                isTerminated = True
    # - - - - - 以上 q7, q8 共通部分 - - - - - - - - - 

    if isEncountered:
        b0_chain[-1].LENGTH = minLength
        b0_chain[-1].END_POS = b0_chain[-1].START_POS + minLength * b0_chain[-1].DIREC
        b0_chain[-1].t = 12 if tgtLinkType in [LinkType.A, LinkType.F] else 3

    return [l, b0_chain]


def q14(l):
    if DebugConst.PRINT_QN:
        print('q14')

    l.t = (l.t) | 2
    l.LINK_TYPE = LinkType.B

    b0_chain = []
    isEncountered = False
    isTerminated = False

    _sucL = Link(linkType=LinkType.B2, startPos=l.END_POS, theta_0=l.THETA + 90, t=0)
    b0_chain.append(_sucL)


    flattenRoadNet = flatten(roadNet, Link)
    
    minLength = 10000
    tgtLinkType = None
    for tgtLink in flattenRoadNet:
        _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
        if _length is None:
            continue
        elif _length > b0_chain[-1].LENGTH * Const.v:
            continue
        elif _length < Const.MIN_LINK_LENGTH:
            continue

        isEncountered = True

        if _length < minLength:
            minLength = _length
            tgtLinkType = tgtLink.LINK_TYPE
        del _length

    while not(isEncountered or isTerminated):

        b0_chain[-1].t = (b0_chain[-1].t) | 2

        _sucL = Link(linkType=LinkType.B2, startPos=b0_chain[-1].END_POS, theta_0=b0_chain[-1].THETA + 90., t=0)
        b0_chain.append(_sucL)

        minLength = 10000
        tgtLinkType = None
        for tgtLink in flattenRoadNet:
            _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
            if _length is None:
                continue
            elif _length > b0_chain[-1].LENGTH * Const.v:
                continue
            elif _length < Const.MIN_LINK_LENGTH:
                continue
            
            isEncountered = True

            if _length < minLength:
                minLength = _length
                tgtLinkType = tgtLink.LINK_TYPE
            del _length

        if not(isEncountered):
            # if not(
            #     Const.Y_LIM[0] <= b0_chain[-1].END_POS[0] <= Const.Y_LIM[1] 
            #     and Const.X_LIM[0] <= b0_chain[-1].END_POS[1] <= Const.X_LIM[1]
            # ):
            #     isTerminated = True

            if len(b0_chain) >= 6:
                isTerminated = True

    if isEncountered:
        b0_chain[-1].LENGTH = minLength
        b0_chain[-1].END_POS = b0_chain[-1].START_POS + minLength * b0_chain[-1].DIREC
        b0_chain[-1].t = 12 if tgtLinkType in [LinkType.A, LinkType.F] else 3
        return [l, b0_chain]
    if isTerminated:
        return [l, b0_chain]

def q15(l):
    if DebugConst.PRINT_QN:
        print('q15')

    l.t = (l.t) | 1
    l.LINK_TYPE = LinkType.B

    b0_chain = []
    isEncountered = False
    isTerminated = False

    _sucL = Link(linkType=LinkType.B1, startPos=l.END_POS, theta_0=l.THETA + 270, t=0)
    b0_chain.append(_sucL)


    flattenRoadNet = flatten(roadNet, Link)
    
    minLength = 10000
    tgtLinkType = None
    for tgtLink in flattenRoadNet:
        _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
        if _length is None:
            continue
        elif _length > b0_chain[-1].LENGTH * Const.v:
            continue
        elif _length < Const.MIN_LINK_LENGTH:
            continue

        isEncountered = True

        if _length < minLength:
            minLength = _length
            tgtLinkType = tgtLink.LINK_TYPE
        del _length

    while not(isEncountered or isTerminated):

        b0_chain[-1].t = (b0_chain[-1].t) | 1

        _sucL = Link(linkType=LinkType.B1, startPos=b0_chain[-1].END_POS, theta_0=b0_chain[-1].THETA + 270., t=0)
        b0_chain.append(_sucL)

        minLength = 10000
        tgtLinkType = None
        for tgtLink in flattenRoadNet:
            _length, _ = b0_chain[-1].getLengthEncounteringTo(tgtLink)
            if _length is None:
                continue
            elif _length > b0_chain[-1].LENGTH * Const.v:
                continue
            elif _length < Const.MIN_LINK_LENGTH:
                continue
            
            isEncountered = True

            if _length < minLength:
                minLength = _length
                tgtLinkType = tgtLink.LINK_TYPE
            del _length

        if not(isEncountered):
            # if not(
            #     Const.Y_LIM[0] <= b0_chain[-1].END_POS[0] <= Const.Y_LIM[1] 
            #     and Const.X_LIM[0] <= b0_chain[-1].END_POS[1] <= Const.X_LIM[1]
            # ):
            #     isTerminated = True

            if len(b0_chain) >= 6:
                isTerminated = True

    if isEncountered:
        b0_chain[-1].LENGTH = minLength
        b0_chain[-1].END_POS = b0_chain[-1].START_POS + minLength * b0_chain[-1].DIREC
        b0_chain[-1].t = 12 if tgtLinkType in [LinkType.A, LinkType.F] else 3
        return [l, b0_chain]
    if isTerminated:
        return [l, b0_chain]



# --------------------------------------------------
# 一覧化
# --------------------------------------------------

P = {}

Q = {
    1: {'alpha': 0., 'k': 1., 'replace':q1},    
    2: {'alpha': 0., 'k': 0.05, 'replace':q2},
    3: {'alpha': 0., 'k': 0.05, 'replace':q3},

    4: {'alpha': 0.1, 'k': 1e-5, 'replace':q4},
    # 4: {'alpha': 0.0, 'k': 0.5, 'replace':q4}, # debug 用の嘘。
    5: {'alpha': 0.1, 'k': 1e-5, 'replace':q5},
    6: {'alpha': 0.1, 'k': 1e-5, 'replace':q6},

    7: {'alpha': 0.1, 'k': 1e-4, 'replace':q7},
    8: {'alpha': 0.1, 'k': 1e-4, 'replace':q8},

    9: {'alpha': 0.01, 'k': 1e-6, 'replace':q9},
    10: {'alpha': 0.01, 'k': 1e-6, 'replace':q10},

    11: {'alpha': 0.1, 'k': 1e-4, 'replace':q11},
    12: {'alpha': 0.1, 'k': 1e-4, 'replace':q12},
    13: {'alpha': 0.1, 'k': 1e-4, 'replace':q13},

    14: {'alpha': 0.01, 'k': 0., 'replace':q14},
    15: {'alpha': 0.01, 'k': 0., 'replace':q15},
}

# -------------------------------------------------------
# omega 作り (超簡単)
# -------------------------------------------------------

# 初期 roadNet (omega ともいう)
# roadNet = [
#     # 原点から始まるとする，そして最初は真右を向いているものとする．
#     Link(LinkType.F, startPos=np.array([0., 0.]), theta_0=0., t=0),
#     Link(LinkType.F, startPos=np.array([0., 0.]), theta_0=180., t=0),
# ]

# # # -------------------------------------------------------
# # # omega 作り (普通)
# # # -------------------------------------------------------

roadNet = []

_link10 = Link(LinkType.A, startPos=np.array([0., 0.]), theta_0=0., t=12)
roadNet.append(_link10)

_link21 = Link(LinkType.F, startPos=_link10.END_POS, theta_0=_link10.THETA+270, t=0)
roadNet.append([_link21,])

_link31 = Link(LinkType.A, startPos=_link10.END_POS, theta_0=_link10.THETA+90, t=0)
_link32 = Link(LinkType.F, startPos=_link31.END_POS, theta_0=_link31.THETA, t=0)
roadNet.append([_link31, _link32])

_link11 = Link(LinkType.A, startPos=_link10.END_POS, theta_0=_link10.THETA, t=0)
roadNet.append(_link11)
roadNet.append(q10(_link11)[1]) # 左回りの地区道路閉路を作っておく (Area の初期値になにか欲しいため)

_link12 = Link(LinkType.F, startPos=_link11.END_POS, theta_0=_link11.THETA, t=0)
roadNet.append(_link12)

roadNet = [roadNet, Link(LinkType.F, startPos=np.array([0., 0.]), theta_0=180., t=0)]




# -------------------------------------------------------
# omega 作り (デバッグ用)
# -------------------------------------------------------

# omega = []

# _link = Link(LinkType.A, startPos=np.array([0., 0.]), theta_0=0., t=12)
# omega.append(_link)

# _link2 = Link(LinkType.F, startPos=_link.END_POS, theta_0=_link.THETA+270, t=0)
# # _link3 = Link(LinkType.A, startPos=_link.END_POS, theta_0=_link.THETA+90, t=0)
# # _link4 = Link(LinkType.F, startPos=_link3.END_POS, theta_0=_link3.THETA, t=0)
# omega.append([_link2,])
# # omega.append([_link3, _link4])

# _link = Link(LinkType.A, startPos=_link.END_POS, theta_0=_link.THETA, t=0)
# omega.append(_link)

# # _link = Link(LinkType.F, startPos=_link.END_POS, theta_0=_link.THETA, t=0)
# # omega.append(_link)

# roadNet = omega