# import system library
from datetime import datetime
import random
import json
from numpy import *
import math
import numpy as np

# import user-library
import load_data




routes = ["BCN_BUD",  # route 1
          "BUD_BCN",  # route 2
          "CRL_OTP",  # route 3
          "MLH_SKP",  # route 4
          "MMX_SKP",  # route 5
          "OTP_CRL",  # route 6
          "SKP_MLH",  # route 7
          "SKP_MMX"]  # route 8

def days_between(d1, d2):
    """
    get the days interval between two dates
    :param d1: date1
    :param d2: date2
    :return: days interval
    """
    d1 = datetime.strptime(d1, "%Y%m%d")
    d2 = datetime.strptime(d2, "%Y%m%d")
    return abs((d2 - d1).days)


def remove_duplicates(values):
    """
    remove duplicate value in a list
    :param values: input list
    :return: no duplicate entry list
    """
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

def getPrice(price):
    """
    Get the numeric price in a string format, which contains currency symbol
    :param price:
    :return:
    """
    price = float( filter( lambda x: x in '0123456789.', price) )
    return price

def pickRandomTicket(filePrefix="BCN_BUD", dataset="large data set"):
    """
    pick 50 tickets randomly for one route
    """

    # get the total departure date length in this route
    departureLen = load_data.get_departure_len(filePrefix, dataset)

    totalPrice = 0
    len = 0

    for index in range(departureLen):
        # get the dataset with same departure date
        datas = load_data.load_data_with_departureIndex(index, filePrefix, dataset)
        date = datas[0]["Date"]
        if int(date) < 20160115 and int(date) < 20160220:
            random.shuffle(datas)
            datas = datas[0:50]
            for data in datas:
                totalPrice += getPrice(data["MinimumPrice"])
            len = len+1

    avgPrice = totalPrice * 1.0 / (len*50)

    return avgPrice

def getRandomTicketPriceForAllRoutes():
    for route in routes:
        avgPrice = pickRandomTicket(route)
        with open('randomPrice/randomPrice_{:}.json'.format(route), 'w') as outfile:
            json.dump(avgPrice, outfile)


def pickRandomTicketByNumpy(flightNum):
    evalMatrix = np.load('inputReg/X_test.npy')
    # take the departure date 20 days after the first observed date
    evalMatrix = evalMatrix[np.where(evalMatrix[:, 8]>20)[0], :]
    # take one route
    evalMatrix = evalMatrix[np.where(evalMatrix[:, flightNum]==1)[0], :]

    totalPrice = 0;
    len = 0;
    departureDates = np.unique(evalMatrix[:, 8])
    for departureDate in departureDates:
        tmpMatrix = evalMatrix[np.where(evalMatrix[:, 8]==departureDate)[0], :]
        tmpMatrix = tmpMatrix[:, 12]
        if tmpMatrix.shape[0] > 30:
            np.random.shuffle(tmpMatrix)
            tmpMatrix = tmpMatrix.reshape((tmpMatrix.shape[0], 1))
            totalPrice += np.sum(tmpMatrix[0:30, :])
            len += 30
        else:
            totalPrice += np.sum(tmpMatrix)
            len += tmpMatrix.shape[0]

    avgPrice = totalPrice * 1.0 / len
    return avgPrice




def getRandomTicketPriceForAllRoutesByNumpy():
    for route in range(8):
        avgPrice = pickRandomTicketByNumpy(route)
        print avgPrice
        #with open('randomPrice/randomPrice_{:}.json'.format(route), 'w') as outfile:
            #json.dump(avgPrice, outfile)


def pickMinTicketByNumpy(flightNum):
    evalMatrix = np.load('inputReg/X_train.npy')
    # take the departure date 20 days after the first observed date
    evalMatrix = evalMatrix[np.where(evalMatrix[:, 8]>20)[0], :]
    # take one route
    evalMatrix = evalMatrix[np.where(evalMatrix[:, flightNum]==1)[0], :]

    totalPrice = 0;
    len = 0;
    departureDates = np.unique(evalMatrix[:, 8])
    for departureDate in departureDates:
        tmpMatrix = evalMatrix[np.where(evalMatrix[:, 8]==departureDate)[0], :]
        tmpMatrix = tmpMatrix[:, 12]
        tmpMatrix = tmpMatrix.reshape((tmpMatrix.shape[0], 1))
        totalPrice += tmpMatrix.max()

    avgPrice = totalPrice * 1.0 / departureDates.shape[0]
    return avgPrice




def getMinTicketPriceForAllRoutesByNumpy():
    for route in range(8):
        avgPrice = pickMinTicketByNumpy(route)
        print avgPrice
        #with open('randomPrice/randomPrice_{:}.json'.format(route), 'w') as outfile:
            #json.dump(avgPrice, outfile)




def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")




if __name__ == "__main__":
    getRandomTicketPriceForAllRoutesByNumpy()
    pass