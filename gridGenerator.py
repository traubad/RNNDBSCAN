from random import randint
import csv

#Used for generating grid dataset

def makeGrid(classifier, num_dimensions,num_observations,d,startX=0,startY=0):
    toAdd = [startX, startY, classifier]
    output = [tuple(toAdd)]
    for i in range((num_observations//2)-1):
        toAdd[randint(0,1)] += d + randint(0,2)
        output.append(tuple(toAdd))
    return output

def writeFile(filename, data):
    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        for row in data:
            writer.writerow(row)


def main():
    totalObservations = 2500
    dimensions = 2
    toWrite = makeGrid(classifier = 1,num_dimensions = dimensions, num_observations = totalObservations//2, d = dimensions) + makeGrid(classifier = 2, num_dimensions = dimensions, num_observations = totalObservations//2, d = dimensions//2, startX=3000, startY=3000)
    writeFile("grid.csv",toWrite)



if __name__ == "__main__":
    main()
