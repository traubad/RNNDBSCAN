from scipy.spatial import distance
import pprint
import argparse
from collections import namedtuple
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import time


pp = pprint.PrettyPrinter(indent=4)
UNCLASSIFIED, NOISE, CORE, BOUNDARY = -1, -2, -3, -4
DataPoint = namedtuple('DataPoint',['baseValues','rnnValues','knnValues', 'distanceValues', 'index', 'classifier'])
X = []

def main():
    global X
    args = parseArgs()
    with open(args.filename) as f:
        df = f.readlines()
    if args.verbose: print("Preprocessing Data")
    startTime = time.time()
    features,classList,classDict = preprocess(df, {'tsv':"\t"}.get(args.filename.split(".")[-1],','))
    highestK = -1
    highestVal = -1
    bestCluster = -1
    bestNoise = -1
    aris = []
    if args.verbose: print("Building Distance Matrix")
    distanceMatrix = createDistanceMatrix(features)
    r = args.k
    start = 2 if len(classDict) - (r//2) < 2 else len(classDict) - (r//2)
    end = start+r
    if args.verbose: print("Running first test: "+str(time.time()-startTime)+" seconds")
    for k in range(start, end):
        currentTaskTime = time.time()
        print("\nCurrent K: "+str(k))
        if args.verbose: print("Finding KNN and RNN...")
        X = buildDataSet(k, features, classList, distanceMatrix)
        print("Running DBSCAN...")
        assign = RNN_DBSCAN(X, k)
        clusters = len([i for i in set(assign) if i >= 0])
        nse = assign.count(-2)
        print("Instances of Noise: "+str(nse))
        print("Clusters: "+str(clusters))
        actuals = [x.classifier for x in X]
        
        ari = adjusted_rand_score(actuals,assign)
        print("ARI: "+str(ari))
        aris.append(ari)
        if ari > highestVal:
            highestVal = ari
            highestK = k
            bestCluster = clusters
            bestNoise = nse
        if args.verbose:
            print("Current Task Runtime: "+str(time.time() - currentTaskTime))
            print("Total Runtime: "+str(time.time() - startTime))
            

    if args.verbose: print("\nPlotting Data")

    
    
    plt.plot([i for i in range(start, end)], aris,'ro')
    plt.plot([highestK], [highestVal],'go')
    plt.xlabel("K")
    plt.ylabel("ARI")
    title = "RNNDBSCAN: " + args.filename.split("/")[-1]+" - "+"optimal K:"+str(highestK)
    title += "  ARI:"+str(round(highestVal,3))+"\nClusters: "+str(bestCluster)
    title += "  Noise:"+str(bestNoise)
    
    plt.title(title)
    print("Work Complete! Total runtime: "+str(time.time() - startTime))
    
    plt.show()

def preprocess(data, splitOn):
    classes = 1
    features = []
    classDict = {}
    classList = []
    for line in data:
        line = line.split(splitOn)
        if len(line) > 1:
            features.append([float(i) for i in line[:-1]])
            if line[-1] not in classDict:
                classDict[line[-1]] = classes
                classes += 1
            classList.append(classDict[line[-1]])

    return features,classList,classDict


def createDistanceMatrix(data):
    outMatrix = [[-1 for y in data] for i in data]
    for baseIndex, baseRow in enumerate(data):
        for otherIndex, otherRow in enumerate(data[baseIndex+1:]):
            dst = distance.euclidean(baseRow,otherRow)
            outMatrix[baseIndex][otherIndex+1+baseIndex] = dst
            outMatrix[otherIndex+1+baseIndex][baseIndex] = dst
    return tuple([tuple(row) for row in outMatrix])

def KNN(distanceMatrix,k):
    k += 1
    newMatrix = [sorted([(index,value) for index,value in enumerate(i)], key=lambda x:x[1])[::1][:k] for i in distanceMatrix]
    newMatrix = [[index for index,dist in row] for row in newMatrix]
    newMatrix = [row[1:] for row in newMatrix]
    return tuple([tuple(row) for row in newMatrix])

def RNN(nm):
    R = [[] for i in nm]
    for base, row in enumerate(nm):
        for v in row:
            R[v].append(base)
    return tuple([tuple(row) for row in R])

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="CSV or TSV file for processing. If file extension isn't tsv, it's assumed to be csv.", type=str)
    parser.add_argument("-k", help="how many different k's to run", type=int, default=10)
    parser.add_argument("--verbose", help="Print out additional runtime data", action="store_true")
    return parser.parse_args()

def buildDataSet(k, data, classifierMatrix, distanceMatrix):
    kMatrix = KNN(distanceMatrix,k)
    rMatrix = RNN(kMatrix)
    return tuple([DataPoint(data[i],rMatrix[i],kMatrix[i],distanceMatrix[i],i,classifierMatrix[i]) for i in range(len(data))])


def RNN_DBSCAN(X,k):
    assign = [UNCLASSIFIED for i in range(len(X))]
    cluster = 1

    for i,x in enumerate(X):
        if assign[i] == UNCLASSIFIED:
            if expandCluster(x, cluster, assign, k):
                cluster += 1

    expandClusters(k,assign)

    return assign

def expandCluster(x, cluster, assign, k):
    if(len(x.rnnValues)) < k:
        assign[x.index] = NOISE
        return False

    else:
        seeds = neighborhood(x,k)
        for v in [x] + seeds: assign[v.index] = cluster

        while seeds:
            y = seeds.pop(0)
            if len(y.rnnValues) >= k:
                neighbors = neighborhood(y,k)
                for z in neighbors:
                    if assign[z.index] == UNCLASSIFIED:
                        seeds.append(z)
                        assign[z.index] = cluster
                    elif assign[z.index] == NOISE:
                        assign[z.index] = cluster
        return True

def neighborhood(x,k):
    return [X[val] for val in x.knnValues] + [X[y] for y in x.rnnValues if len(X[y].rnnValues) >= k]

def expandClusters(k,assign):
    for x in X:
        if assign[x.index] == NOISE:
            neighbors = x.knnValues
            mincluster = NOISE
            mindist = -1

            for i in neighbors:
                n = X[i]
                cluster = assign[i]
                d = x.distanceValues[i]

                if len(n.knnValues) >= k and d <= density(cluster,assign) and (d < mindist or mindist == -1):
                    mincluster = cluster
                    mindist = d
            assign[x.index] = mincluster

def density(cluster, assign):
    clusterPoints = [i for i in range(len(assign)) if assign[i] == cluster]
    maxDist = -1
    for baseIndex, baseRow in enumerate(clusterPoints):
        for otherIndex, otherRow in enumerate(clusterPoints[baseIndex+1:]):
            if maxDist < X[baseRow].distanceValues[otherRow]:
                maxDist = X[baseRow].distanceValues[otherRow]

    return maxDist

##def purity(X, assign):
##    clusters = [i for i in set(assign) if i >= 0]
##    for cluster in clusters:
##        counts = {}
##        for i,v in enumerate(assign):
##            if v == cluster:
##                if X[i].classifier in counts:
##                    counts[X[i].classifier] += 1
##                else:
##                    counts[X[i].classifier] = 1
##
##        biggestI, biggestV = -1,-1
##        total = 0
##        for key in counts:
##            total += counts[key]
##            if counts[key] > biggestV:
##                biggestI = key
##                biggestV = counts[key]
##                
##        return float(biggestV)/total
##                        
            


if __name__ == '__main__':
    main()
