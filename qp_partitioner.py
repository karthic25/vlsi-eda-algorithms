import sys
from collections import defaultdict
from random import randint
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readInput(ifile):
    with open(ifile, 'r') as f:
        lines = f.readlines()

    nGates, nNets = list(map(int, lines[0].split()))

    net2Gates = defaultdict(list)
    for i in range(1, nGates+1):
        line = list(map(int, lines[i].split()))
        gateId, nNetsConn = line[0] - 1, line[1]
        for j in range(2, nNetsConn+2):
            netId = line[j] - 1
            net2Gates[netId].append(gateId)
    
    net2GatesList = []
    for i in range(nNets):
        net2GatesList.append(net2Gates[i])
        
    nPads = int(lines[nGates+1])
    pad2Nets, padCoords = defaultdict(list), []
    for i in range(nGates+2, nGates+2+nPads):
        line = lines[i].split()
        padId, netId, x, y = int(line[0])-1, int(line[1])-1, float(line[2]), float(line[3])
        pad2Nets[padId].append(netId)
        padCoords.append([x, y])
    padCoords = np.array(padCoords)

    pad2NetsList = []
    for i in range(nPads):
        pad2NetsList.append(pad2Nets[i])
        
    return net2GatesList, padCoords, pad2NetsList, nGates

def annotateMarks(coords, s):
    for i in range(1, len(coords)+1):
        plt.annotate(f'{s} {i}', coords[i-1])
        
def plotCoords(coords, min_=0, max_=100):
    plt.scatter(coords[:,0], coords[:,1], marker='s')
    plt.plot([min_,min_], [min_,max_], color='black')
    plt.plot([min_,max_], [min_,min_], color='black')
    plt.plot([min_,max_], [max_,max_], color='black')
    plt.plot([max_,max_], [max_,min_], color='black')
    plt.grid(visible=True)
    annotateMarks(coords, 'Pad')

def plotGateCoords(net2Gates, padCoords, pad2Nets, nGates, gateCoords=None, annotate=False):
    if gateCoords is None:
        gateCoords = [[randint(0, 101), randint(0, 101)] for i in range(nGates)]
    gateCoords = np.array(gateCoords)
    plt.scatter(gateCoords[:,0], gateCoords[:,1], marker='^', facecolor='green')
    
    for padId in range(len(padCoords)):
        for netId in pad2Nets[padId]:
            for gateId in net2Gates[netId]:
                plt.plot([padCoords[padId][0], gateCoords[gateId][0]], [padCoords[padId][1], gateCoords[gateId][1]], c='red')
    
    for netId in range(len(net2Gates)):
        gates = net2Gates[netId]
        for i in range(len(gates)):
            for j in range(i+1, len(gates)):
                plt.plot([gateCoords[gates[i]][0], gateCoords[gates[j]][0]], [gateCoords[gates[i]][1], gateCoords[gates[j]][1]], c='orange')
    if annotate:
        annotateMarks(gateCoords, '')

def getWeights(net2Gates, padCoords, pad2Nets, nGates):
    weights = [0]*len(net2Gates)
    
    for netId in range(len(net2Gates)):
        gates = net2Gates[netId]
        weights[netId] = len(gates) - 1
        
    for padId in range(len(pad2Nets)):
        for netId in pad2Nets[padId]:
            weights[netId] += 1
    
    weights = [1/w if w > 0 else np.nan for w in weights]
    return np.array(weights)

def getGate2Nets(net2Gates):
    gate2Nets = defaultdict(list)
    for netId in range(len(net2Gates)):
        gates = net2Gates[netId]
        for gateId in gates:
            gate2Nets[gateId].append(netId)
    
    gate2NetsList = []
    for gateId in range(len(gate2Nets)):
        gate2NetsList.append(gate2Nets[gateId])
        
    return gate2NetsList

def getC(weights, net2Gates, padCoords, pad2Nets, nGates):
    C = np.array([[0.0]*nGates]*nGates)
    for netId in range(len(net2Gates)):
        gates = net2Gates[netId]
        for i in range(len(gates)):
            for j in range(i+1, len(gates)):
                C[gates[i], gates[j]] += weights[netId]
                C[gates[j], gates[i]] += weights[netId]
    return C

def getA(C, weights, net2Gates, padCoords, pad2Nets, nGates):
    A = -C
    for gateId in range(nGates):
        A[gateId][gateId] += C[gateId, :].sum()
    
    for padId in range(len(pad2Nets)):
        for netId in pad2Nets[padId]:
            gates = net2Gates[netId]
            for gateId in gates:
                A[gateId][gateId] += weights[netId]
    return A

def getB(weights, net2Gates, padCoords, pad2Nets, nGates):
    b = np.array([[0.0, 0.0]]*nGates)
    for padId in range(len(pad2Nets)):
        for netId in pad2Nets[padId]:
            gates = net2Gates[netId]
            for gateId in gates:
                b[gateId][0] += padCoords[padId][0]*weights[netId]
                b[gateId][1] += padCoords[padId][1]*weights[netId]
    return b

def optimizeQL(weights, net2Gates, padCoords, pad2Nets, nGates):
    C = getC(weights, net2Gates, padCoords, pad2Nets, nGates)
    A = getA(C, weights, net2Gates, padCoords, pad2Nets, nGates)
    b = getB(weights, net2Gates, padCoords, pad2Nets, nGates)
    x = np.linalg.solve(A, b[:, 0])
    y = np.linalg.solve(A, b[:, 1])
    gateCoords = np.transpose(np.array([x, y]))
    return gateCoords
    
def getQL(x, y, w):
    return w*((x[0]-x[1])**2 + (y[0]-y[1])**2)

def getHPWL(x, y, w):
    return w*(abs(x[0]-x[1]) + abs(y[0]-y[1]))

def calculateQLAndHPWL(weights, net2Gates, padCoords, pad2Nets, nGates, gateCoords=None):
    QL, HPWL = 0.0, 0.0
    if gateCoords is None:
        gateCoords = [[randint(0, 101), randint(0, 101)] for i in range(nGates)]
    gateCoords = np.array(gateCoords)
    
    for padId in range(len(padCoords)):
        for netId in pad2Nets[padId]:
            for gateId in net2Gates[netId]:
                QL += getQL([padCoords[padId][0], gateCoords[gateId][0]], [padCoords[padId][1], gateCoords[gateId][1]], weights[netId])
                HPWL += getHPWL([padCoords[padId][0], gateCoords[gateId][0]], [padCoords[padId][1], gateCoords[gateId][1]], weights[netId])
    
    for netId in range(len(net2Gates)):
        gates = net2Gates[netId]
        for i in range(len(gates)):
            for j in range(i+1, len(gates)):
                QL += getQL([gateCoords[gates[i]][0], gateCoords[gates[j]][0]], [gateCoords[gates[i]][1], gateCoords[gates[j]][1]], weights[netId])
                HPWL += getHPWL([gateCoords[gates[i]][0], gateCoords[gates[j]][0]], [gateCoords[gates[i]][1], gateCoords[gates[j]][1]], weights[netId])
    return round(QL, 2), round(HPWL, 2)

def assignment(gateCoords, nGates):
    gateCoords = np.c_[gateCoords, range(nGates)]
    gates = np.array(sorted(gateCoords, key=tuple))[:,2]
    return np.vectorize(int)(gates[:nGates//2]), np.vectorize(int)(gates[nGates//2:])

def getMap(gates):
    gateMap = defaultdict(lambda: -1)
    for i in range(len(gates)):
        gateMap[gates[i]] = i
    return gateMap

def leftContainment(gateCoords, net2Gates, padCoords, pad2Nets, gates, gateMap):
    net2Gates = deepcopy(net2Gates)
    padCoords = deepcopy(padCoords)
    pad2Nets = deepcopy(pad2Nets)
    
    for i in range(len(padCoords)):
        if padCoords[i,0] > 50:
            padCoords[i,0] = 50

    gate2Pad = defaultdict(lambda: -1)
    for netId in range(len(net2Gates)):
        for gateId in net2Gates[netId]:
            if gateMap[gateId] != -1: continue
            if gate2Pad[gateId] != -1: 
                pad2Nets[gate2Pad[gateId]].append(netId)
            else:
                padCoords = np.r_[padCoords, [[50.0, gateCoords[gateId, 1]]]]
                pad2Nets.append([netId])
                gate2Pad[gateId] = len(pad2Nets)-1
        net2Gates[netId] = [gateMap[gateId] for gateId in net2Gates[netId] if gateMap[gateId] != -1]
    
    return net2Gates, padCoords, pad2Nets

def updateGateCoordsLeft(leftMap, gateCoordsLeft, gateCoords):
    gateCoordsUpdated = []
    for i in range(len(gateCoords)):
        if leftMap[i] != -1:
            gateCoordsUpdated.append(gateCoordsLeft[leftMap[i]])
        else:
            gateCoordsUpdated.append(gateCoords[i])
    return np.array(gateCoordsUpdated)

def rightContainment(gateCoords, net2Gates, padCoords, pad2Nets, gates, gateMap):
    net2Gates = deepcopy(net2Gates)
    padCoords = deepcopy(padCoords)
    pad2Nets = deepcopy(pad2Nets)
    
    for i in range(len(padCoords)):
        if padCoords[i,0] < 50:
            padCoords[i,0] = 50

    gate2Pad = defaultdict(lambda: -1)
    for netId in range(len(net2Gates)):
        for gateId in net2Gates[netId]:
            if gateMap[gateId] != -1: continue
            if gate2Pad[gateId] != -1: 
                pad2Nets[gate2Pad[gateId]].append(netId)
            else:
                padCoords = np.r_[padCoords, [[50.0, gateCoords[gateId, 1]]]]
                pad2Nets.append([netId])
                gate2Pad[gateId] = len(pad2Nets)-1
        net2Gates[netId] = [gateMap[gateId] for gateId in net2Gates[netId] if gateMap[gateId] != -1]
    
    return net2Gates, padCoords, pad2Nets

def updateGateCoordsRight(rightMap, gateCoordsRight, gateCoords):
    gateCoordsUpdated = []
    for i in range(len(gateCoords)):
        if rightMap[i] != -1:
            gateCoordsUpdated.append(gateCoordsRight[rightMap[i]])
        else:
            gateCoordsUpdated.append(gateCoords[i])
    return np.array(gateCoordsUpdated)

def _3QP(net2Gates, padCoords, pad2Nets, nGates):
    # QP 1
    weights = getWeights(net2Gates, padCoords, pad2Nets, nGates)
    gateCoords = optimizeQL(weights, net2Gates, padCoords, pad2Nets, nGates)
    
    # Assignment
    left, right = assignment(gateCoords, nGates)
    left, right = np.vectorize(int)(left), np.vectorize(int)(right)
    leftMap, rightMap = getMap(left), getMap(right)
    
    # Left Containment
    net2GatesLeft, padCoordsLeft, pad2NetsLeft = leftContainment(gateCoords, net2Gates, padCoords, pad2Nets, left, leftMap)
    
    # QP 2
    weightsLeft = getWeights(net2GatesLeft, padCoordsLeft, pad2NetsLeft, len(left))
    gateCoordsLeft = optimizeQL(weightsLeft, net2GatesLeft, padCoordsLeft, pad2NetsLeft, len(left))
    gateCoordsUpdated = updateGateCoordsLeft(leftMap, gateCoordsLeft, gateCoords)
    
    # Right Containment
    net2GatesRight, padCoordsRight, pad2NetsRight = rightContainment(gateCoordsUpdated, net2Gates, padCoords, pad2Nets, right, rightMap)
    
    # QP 3
    weightsRight = getWeights(net2GatesRight, padCoordsRight, pad2NetsRight, len(right))
    gateCoordsRight = optimizeQL(weightsRight, net2GatesRight, padCoordsRight, pad2NetsRight, len(right))
    gateCoordsUpdated2 = updateGateCoordsRight(rightMap, gateCoordsRight, gateCoordsUpdated)
    print('optimized QL & HPWL:', calculateQLAndHPWL(weights, net2Gates, padCoords, pad2Nets, nGates, gateCoords=gateCoordsUpdated2))
    
    # plotCoords(padCoords, min_=0, max_=100)
    # plotGateCoords(net2Gates, padCoords, pad2Nets, nGates, gateCoords=gateCoordsUpdated2, annotate=True)
    plt.show()
    return gateCoordsUpdated2

def write_coords(ofile, coords):
    coords = np.around(coords, decimals=8)
    with open(ofile, 'w') as f:
        count = 1
        for x, y in coords:
            f.write(f'{count} {x} {y}\n')
            count += 1

def main():
	folder_map = {
	    "3QP": ("primary1", "struct", "fract", "toy1", "toy2"),
	    # "8x8 QP": ("biomed", "industry1", "industry2")
	}
	for k, values in folder_map.items():
	    for v in values:
	        print(k, v)
	        net2Gates, padCoords, pad2Nets, nGates = readInput(f"../../benchmarks/{k}/{v}")
	        gateCoordsUpdated2 = _3QP(net2Gates, padCoords, pad2Nets, nGates)
	        write_coords(f"../../benchmark solutions/{k}/{v}", gateCoordsUpdated2)

if __name__ == "__main__":
	main()