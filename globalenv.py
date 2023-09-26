from sys import stdout

from pypower.api import runopf
from numpy import array, sum
import os
from numpy import ones, zeros, r_, sort, exp, pi, diff, arange, real, imag

from numpy import flatnonzero as find

from pypower.idx_bus import BUS_I, PD, QD, GS, BS, BUS_AREA, VM, VA
from pypower.idx_gen import QG, QMAX, QMIN, GEN_STATUS
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS

from pypower.isload import isload
from pypower.ppoption import ppoption
import numpy as np
from case15da import case15da
import copy

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class globalenv:
    ppopt = ppoption(PF_ALG=1, OUT_GEN=1)
    def __init__(self, client_num):
        self.case = case15da()
        self.client_num = client_num
        self.client_action = []
        for i in range(self.client_num):
            self.client_action.append([0, 0])
        self.sumP = self.substationP(self.case)
    # 计算线路损失
    def calloss(self, baseMVA, bus=None, gen=None, branch=None, f=None, success=None, t=None, fd=None, ppopt=None):
        if isinstance(baseMVA, dict):
            have_results_struct = 1
            results = baseMVA
            if gen is None:
                ppopt = ppoption()  ## use default options
            else:
                ppopt = gen
            if (ppopt['OUT_ALL'] == 0):
                return  ## nothin' to see here, bail out now
            if bus is None:
                fd = stdout  ## print to stdout by default
            else:
                fd = bus
            baseMVA, bus, gen, branch, success, et = \
                results["baseMVA"], results["bus"], results["gen"], \
                results["branch"], results["success"], results["et"]
            if 'f' in results:
                f = results["f"]
            else:
                f = None
        else:
            have_results_struct = 0
            if ppopt is None:
                ppopt = ppoption()  ## use default options
                if fd is None:
                    fd = stdout  ## print to stdout by default
            if ppopt['OUT_ALL'] == 0:
                return  ## nothin' to see here, bail out now

        isOPF = f is not None  ## FALSE -> only simple PF data, TRUE -> OPF data

        ## options
        isDC = ppopt['PF_DC']  ## use DC formulation?
        OUT_ALL = ppopt['OUT_ALL']
        OUT_ANY = OUT_ALL == 1  ## set to true if any pretty output is to be generated
        OUT_SYS_SUM = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_SYS_SUM'])
        OUT_AREA_SUM = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_AREA_SUM'])
        OUT_BUS = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_BUS'])
        OUT_BRANCH = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_BRANCH'])
        OUT_GEN = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_GEN'])
        OUT_ANY = OUT_ANY | ((OUT_ALL == -1) and
                             (OUT_SYS_SUM or OUT_AREA_SUM or OUT_BUS or
                              OUT_BRANCH or OUT_GEN))

        if OUT_ALL == -1:
            OUT_ALL_LIM = ppopt['OUT_ALL_LIM']
        elif OUT_ALL == 1:
            OUT_ALL_LIM = 2
        else:
            OUT_ALL_LIM = 0

        OUT_ANY = OUT_ANY or (OUT_ALL_LIM >= 1)
        if OUT_ALL_LIM == -1:
            OUT_V_LIM = ppopt['OUT_V_LIM']
            OUT_LINE_LIM = ppopt['OUT_LINE_LIM']
            OUT_PG_LIM = ppopt['OUT_PG_LIM']
            OUT_QG_LIM = ppopt['OUT_QG_LIM']
        else:
            OUT_V_LIM = OUT_ALL_LIM
            OUT_LINE_LIM = OUT_ALL_LIM
            OUT_PG_LIM = OUT_ALL_LIM
            OUT_QG_LIM = OUT_ALL_LIM

        OUT_ANY = OUT_ANY or ((OUT_ALL_LIM == -1) and (OUT_V_LIM or OUT_LINE_LIM or OUT_PG_LIM or OUT_QG_LIM))
        ptol = 1e-4  ## tolerance for displaying shadow prices

        ## create map of external bus numbers to bus indices
        i2e = bus[:, BUS_I].astype(int)
        e2i = zeros(max(i2e) + 1, int)
        e2i[i2e] = arange(bus.shape[0])

        ## sizes of things
        nb = bus.shape[0]  ## number of buses
        nl = branch.shape[0]  ## number of branches
        ng = gen.shape[0]  ## number of generators

        ## zero out some data to make printout consistent for DC case
        if isDC:
            bus[:, r_[QD, BS]] = zeros((nb, 2))
            gen[:, r_[QG, QMAX, QMIN]] = zeros((ng, 3))
            branch[:, r_[BR_R, BR_B]] = zeros((nl, 2))

        ## parameters
        ties = find(bus[e2i[branch[:, F_BUS].astype(int)], BUS_AREA] !=
                    bus[e2i[branch[:, T_BUS].astype(int)], BUS_AREA])
        ## area inter-ties
        tap = ones(nl)  ## default tap ratio = 1 for lines
        xfmr = find(branch[:, TAP])  ## indices of transformers
        tap[xfmr] = branch[xfmr, TAP]  ## include transformer tap ratios
        tap = tap * exp(-1j * pi / 180 * branch[:, SHIFT])  ## add phase shifters
        nzld = find((bus[:, PD] != 0.0) | (bus[:, QD] != 0.0))
        sorted_areas = sort(bus[:, BUS_AREA])
        ## area numbers
        s_areas = sorted_areas[r_[1, find(diff(sorted_areas)) + 1]]
        nzsh = find((bus[:, GS] != 0.0) | (bus[:, BS] != 0.0))
        allg = find(~isload(gen))
        ong = find((gen[:, GEN_STATUS] > 0) & ~isload(gen))
        onld = find((gen[:, GEN_STATUS] > 0) & isload(gen))
        V = bus[:, VM] * exp(-1j * pi / 180 * bus[:, VA])
        out = find(branch[:, BR_STATUS] == 0)  ## out-of-service branches
        nout = len(out)
        if isDC:
            loss = zeros(nl)
        else:
            loss = baseMVA * abs(V[e2i[branch[:, F_BUS].astype(int)]] / tap -
                                 V[e2i[branch[:, T_BUS].astype(int)]]) ** 2 / \
                   (branch[:, BR_R] - 1j * branch[:, BR_X])

        fchg = abs(V[e2i[branch[:, F_BUS].astype(int)]] / tap) ** 2 * branch[:, BR_B] * baseMVA / 2
        tchg = abs(V[e2i[branch[:, T_BUS].astype(int)]]) ** 2 * branch[:, BR_B] * baseMVA / 2
        loss[out] = zeros(nout)
        fchg[out] = zeros(nout)
        tchg[out] = zeros(nout)
        # print(sum(real(loss)), sum(imag(loss)))
        return sum(real(loss)), sum(imag(loss))
    # 计算substation的有功功率
    def substationP(self, data):
        # with suppress_stdout_stderr():
        result = runopf(data, self.ppopt)
        loss = self.calloss(result, self.ppopt)
        return loss[0] + sum(data['bus'][:, 2])
    # 计算全局的回报，其中bus_list表示每个EV连接的bus序号，clients_power_list表示所有agent一轮的所有action，time_list表示所有agent一轮采取动作的时间点集合
    def calculateReward(self, bus_list, clients_power_list, clients_time_list, kw=20):
        time_power_reward = [None] * 24
        for h in range(24):
            case_copy = copy.deepcopy(self.case)
            judge = False
            for client_id in range(len(bus_list)):
                client_time_list = clients_time_list[client_id]
                client_power_list = clients_power_list[client_id]
                for i in range(len(client_power_list)):
                    time = client_time_list[i]
                    if time % 24 == h:  # 如果已经过了24小时，计算一次OPF
                        judge = True
                        case_copy['bus'][bus_list[client_id]-1][2] += client_power_list[i]  # 让EV所连的对应的bus的有功功率加减EV的功率
            diff = 0.0
            latestP = 0.0
            if judge:
                latestP = self.substationP(case_copy)
                # 标准差
                # diff = abs(self.sumP - latestP) * 10.0 * kw
            # time_power_reward[h] = np.array([-diff], dtype='f8')
            time_power_reward[h] = np.array([-abs(latestP)*2], dtype='f8')
        return time_power_reward

    def getSubstationPList(self, bus_list, EVs_power_list):
        substationP_list = []
        for j in range(len(EVs_power_list[0])):
            case_copy = copy.deepcopy(self.case)
            # originP = self.substationP(case_copy)
            for i in range(len(bus_list)):
                case_copy['bus'][bus_list[i] - 1][2] += EVs_power_list[i][j] * 0.15
            latestP = self.substationP(case_copy)
            substationP_list.append((latestP) * 100.0)
        return substationP_list


















