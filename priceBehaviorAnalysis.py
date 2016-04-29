# system library
import numpy as np
import pandas as pd
from matplotlib.dates import MONDAY
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter


import matplotlib.pyplot as plt


# route prefix
routes = ["BCN_BUD",  # route 1
          "BUD_BCN",  # route 2
          "CRL_OTP",  # route 3
          "MLH_SKP",  # route 4
          "MMX_SKP",  # route 5
          "OTP_CRL",  # route 6
          "SKP_MLH",  # route 7
          "SKP_MMX"]  # route 8

# feature 0~7: flight number dummy variables
# feature 8: departure date; feature 9: observed date state;
# feature 10: minimum price; feature 11: maximum price
# feature 12: current price
train = np.load('Regression/inputReg/X_train.npy')
test = np.load('Regression/inputReg/X_test.npy')
datas = np.concatenate((train, test), axis=0)

def getOneRouteData(X, route):
    # route index
    flightNum = routes.index(route)

    # choose one route datas
    X = X[np.where(X[:, flightNum]==1)[0], :]

    return X

def getMinAndMaxForOneRoute(route):
    X = getOneRouteData(datas, route)

    min = np.amin(X[:,12])
    max = np.amax(X[:,12])

    return min, max

def MinMaxComparison():
    """
    Do the comparison of the max and min price for each route
    :return: NA
    """
    for i in range(8):
        print routes[i]
        min, max = getMinAndMaxForOneRoute(routes[i])
        print min
        print max
        print max-min
        print "\n"

def getDatasForOneRouteForOneDepartureDate(route, departureDate):
    X = getOneRouteData(datas, route)
    minDeparture = np.amin(X[:,8])
    maxDeparture = np.amax(X[:,8])
    print minDeparture
    print maxDeparture

    # get specific departure date datas
    X = X[np.where(X[:, 8]==departureDate)[0], :]

    # get the x values
    xaxis = X[:,9] # observed date state
    print xaxis
    xaxis = departureDate-1-xaxis
    print xaxis

    tmp = xaxis
    startdate = "20151109"
    xaxis = [pd.to_datetime(startdate) + pd.DateOffset(days=state) for state in tmp]
    print xaxis

    # get the y values
    yaxis = X[:,12]


    # every monday
    mondays = WeekdayLocator(MONDAY)

    # every 3rd month
    months = MonthLocator(range(1, 13), bymonthday=1, interval=01)
    days = WeekdayLocator(byweekday=1, interval=2)
    monthsFmt = DateFormatter("%b. %d, %Y")

    fig, ax = plt.subplots()
    ax.plot_date(xaxis, yaxis, 'r--')
    ax.plot_date(xaxis, yaxis, 'bo')
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(monthsFmt)
    #ax.xaxis.set_minor_locator(mondays)
    ax.autoscale_view()
    #ax.xaxis.grid(False, 'major')
    #ax.xaxis.grid(True, 'minor')
    ax.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Price in Euro')

    fig.autofmt_xdate()
    plt.show()

    """
    # plot
    line1, = plt.plot(xaxis, yaxis, 'r--')
    line2, = plt.plot(xaxis, yaxis, 'bo')
    #plt.legend([line2], ["Price"])
    plt.xlabel('States')
    plt.ylabel('Price in Euro')
    plt.show()
    """





if __name__ == "__main__":
    # route 0, 56
    # route 0, 59, rise before departure!
    # bcn-bud 48, 49
    # bud-bcn 30, 49
    # crl-otp 38, 42, 44!
    getDatasForOneRouteForOneDepartureDate(routes[2], 44)

