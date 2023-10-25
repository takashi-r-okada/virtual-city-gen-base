import numpy as np 
# from pathlib import Path 
# import shutil
# import cv2
import matplotlib.pyplot as plt
# from enum import Enum
# import math
# import random
from lib_const import Const
from lib_link import LinkType, Link, roadNet, flatten, areas
from datetime import datetime
import sys
from pprint import pprint

#TODO area の初期状態を作成
#TODO area は地区道路で作れるので、そういう風に変更
#TODO area の時間発展を記述

# -------------------------------------------------------
# 幾何関数
# -------------------------------------------------------

def getGC(vertices: list):
    """
    多角形の重心を求める
    vertices は pos (np.ndarray) がたくさんはいったやつ
    """

    assert len(vertices) >= 3

    subGCs = []
    subSquares = []

    vertices[0]
    for i in range(1, len(vertices)-1):
        subGCs.append((vertices[0] + vertices[i] + vertices[i+1]) / 3.)
        subSquares.append(np.cross(vertices[i] - vertices[0], vertices[i+1] - vertices[0]) / 2.)

    subGCs = np.array(subGCs)
    subSquares = np.array(subSquares)

    return (subSquares.T @ subGCs).T / np.sum(subGCs)



def posIsInLim(pos: np.ndarray):
    return (
        (Const.Y_LIM[0] <= pos[0] <= Const.Y_LIM[1])
        and (Const.X_LIM[0] <= pos[1] <= Const.X_LIM[1])
    )



# -------------------------------------------------------
# area (発展度 area.G、発展度場 area.getT())
# -------------------------------------------------------

class Area():
    """
    エリアオブジェクトのクラス．
    リンクの閉路によって定義される．
    発展度 G を持つ．また，中心座標がしばしば利用される．
    発展度はエリアの単位面積あたりの値であるとする

    4 章によると、地区道路の閉路も area を構成できる
    """
    def __init__(self, links: list):
        self.LINKS = links

        # エリアの中心座標は不変なので定数として持つ．
        self.GC = getGC(list(map(lambda link: link.START_POS, self.LINKS)))

        # 発展度 G は変数．
        self.G = 0.

    def getT(self):
        """
        END_POS (屈折点・交差点) の発展度場 T の値。
        G_i を決定する引数ともなる

        cf. Link.getT
        """

        T_candidates = []
        for a in areas:
            if a is self:
                continue
            else:
                T_candidates.append(a.G * (1. - ((self.END_POS - a.GC)/Const.D_MAX) ** Const.NU))
        T = max(T_candidates)
        T = np.clip(T, 0., 1.)
        return T


    def step(self):
        pass #TODO

# ********** エリアの初期状態定義 (omega 作り (普通) で omega を作った場合) **********


# pprint(roadNet, indent=2)
area = Area(links=[
    roadNet[0][3],
    roadNet[0][4][0],
    roadNet[0][4][1],
    #TODO ここでぶつかられた時に link を分割しないといけなくなった。先にそっちを実装。
])
areas.append(area)





# -------------------------------------------------------
# 時間発展関数 
# stepRoadNet() を一度実行すると、必ず regularizeRoadNet() を実行する
# -------------------------------------------------------

encountedLinkPairs = []

def stepRoadNet(roadNet):
    """
    時間発展
    """
    # print("[stepRoadNet] 開始", roadNet)
    for r in range(len(roadNet)):
        # print("[stepRoadNet] 現在:", roadNet[r])
        if type(roadNet[r]) == Link:
            # print(roadNet[r].LINK_TYPE)
            if posIsInLim(roadNet[r].END_POS) and posIsInLim(roadNet[r].START_POS):
                roadNet[r] = roadNet[r].getStepped()
            else:
                roadNet[r] = [roadNet[r], ]
        else:
            roadNet[r] = stepRoadNet(roadNet[r])
            # print(r, roadNet[r])
    # print("[stepRoadNet] sum前", roadNet)

    ret = []
    for n in roadNet:
        ret += n

    # print("[stepRoadNet] sum後", ret)
    return ret

def regularizeRoadNet(roadNet):
    """
    リンクの分割が必要でまだ未分割なところを分割する
    uLink とはぶつかった側、vLink とはぶつかられた側。
    また、便宜上新しく wLink を置く (vLink の新交差点より先のこと))

    方針: 
    - uLink はここでは変更しない。変更の必要がないから
    - vLink.END_POS は新交差点まで縮められる
    - vLink.t は uLink がぶつかってきた側を考慮して更新
    - wLink を作成 (vLink より)
    - roadNet の vLink, が入っていたところを vLink, wLink, とする


    - 古い area 1つを消して、新しい area 2 つを areas に加える。発展度は 2 つの area で同一とする。
    """

    for uLink, vLink in encountedLinkPairs:
        pass #TODO

    return roadNet




# -------------------------------------------------------
# 初期状態の定義
# -------------------------------------------------------

if __name__ == '__main__':

    for i in range(Const.EPOCH):
        print("\r%d" % i, end="")
        encountedLinkPairs = []
        stepRoadNet(roadNet)
        regularizeRoadNet(roadNet)

        # --------------------------------------------------
        # 観望用マップの出力
        # --------------------------------------------------

        # if i % 1 == 0:
        if i == Const.EPOCH - 1:
            yLim = [-1, 1]
            xLim = [-1, 1]

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(1,1,1)
            
            for link in flatten(roadNet, Link):

                if link.LINK_TYPE in [LinkType.F, LinkType.A]:
                    continue

                lineWidth = 3.5
                clr = 'gray'
                plt.plot(
                    [link.START_POS[1], link.END_POS[1]],
                    [link.START_POS[0], link.END_POS[0]],
                    '-k',
                    color=clr,
                    linewidth=lineWidth
                )
    
            for link in flatten(roadNet, Link):

                if not(link.LINK_TYPE in [LinkType.F, LinkType.A]):
                    continue

                lineWidth = 5
                clr = 'green'
                plt.plot(
                    [link.START_POS[1], link.END_POS[1]],
                    [link.START_POS[0], link.END_POS[0]],
                    '-k',
                    color=clr,
                    linewidth=lineWidth
                )

                if yLim[0] > min(link.START_POS[0], link.END_POS[0]):
                    yLim[0] = min(link.START_POS[0], link.END_POS[0])
                if yLim[1] < max(link.START_POS[0], link.END_POS[0]):
                    yLim[1] = max(link.START_POS[0], link.END_POS[0])
                if xLim[0] > min(link.START_POS[1], link.END_POS[1]):
                    xLim[0] = min(link.START_POS[1], link.END_POS[1])
                if xLim[1] < max(link.START_POS[1], link.END_POS[1]):
                    xLim[1] = max(link.START_POS[1], link.END_POS[1])

            for link in flatten(roadNet, Link):

                if link.LINK_TYPE in [LinkType.F, LinkType.A]:
                    continue

                lineWidth = 1.5
                clr = 'lightgrey'
                plt.plot(
                    [link.START_POS[1], link.END_POS[1]],
                    [link.START_POS[0], link.END_POS[0]],
                    '-k',
                    color=clr,
                    linewidth=lineWidth
                )



            for link in flatten(roadNet, Link):

                if not(link.LINK_TYPE in [LinkType.F, LinkType.A]):
                    continue

                lineWidth = 2.5
                clr = 'snow'
                plt.plot(
                    [link.START_POS[1], link.END_POS[1]],
                    [link.START_POS[0], link.END_POS[0]],
                    '-k',
                    color=clr,
                    linewidth=lineWidth
                )
    

            plotSize = min(yLim[1] - yLim[0], xLim[1] - xLim[0], Const.Y_LIM[1] - Const.Y_LIM[0], Const.X_LIM[1] - Const.X_LIM[0]) * 0.9
            ax.set_ylim((yLim[0] + yLim[1])/2. - plotSize/2., (yLim[0] + yLim[1])/2. + plotSize/2.)
            ax.set_xlim((xLim[0] + xLim[1])/2. - plotSize/2., (xLim[0] + xLim[1])/2. + plotSize/2.)

            ax.set_facecolor('ivory')
            ax.set_aspect("equal", adjustable="box")
            
            ax.grid(alpha=0.4)
            # plt.show()

            plt.savefig(r"C:\Users\okada\Desktop\cellam\city-generation-base\resultOut" + "\\" + datetime.now().strftime("%y%m%dT%H%M") + ".png")
            plt.show()
