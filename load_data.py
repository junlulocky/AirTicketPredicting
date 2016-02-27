# system-library
import json
import os

# user-library
import util as util

routes = ["BCN_BUD",  # route 1
          "BUD_BCN",  # route 2
          "CRL_OTP",  # route 3
          "MLH_SKP",  # route 4
          "MMX_SKP",  # route 5
          "OTP_CRL",  # route 6
          "SKP_MLH",  # route 7
          "SKP_MMX"]  # route 8


def is_not_nullprice(data):
    """
    used by the filter to filter out the null entries
    :param data: input data
    :return: true if it's not null, false if null
    """
    return data and data["MinimumPrice"] != None

def check_if_only_one_flightNum(datas):
    """
    check whether the datas only contain one flight number
    :param datas: input data
    :return: Ture if the datas only contain one flight number, False otherwise
    """
    kinds = []
    for data in datas:
        kinds += data["Flights"]

    flightNums = []
    for kind in kinds:
        flightNums.append(kind["FlightNumber"])

    if len(util.remove_duplicates(flightNums)) == 1:
        return True
    else:
        return False



def load_data_with_prefix_and_dataset(filePrefix="BCN_BUD", dataset="large data set"):
    currentDir = os.path.dirname(os.path.realpath(__file__))
    observeDatesDirs = os.listdir(currentDir + "/data/" + dataset) # path directory of each observed date in the dataset

    filePaths = [] # keep all the file paths start with "filePrefix"
    data_decoded = [] # keep all the schedules start with "filePrefix"

    for date in observeDatesDirs:
        currentPath = currentDir + "/data/" + dataset + "/" + date

        try:
            files = os.listdir(currentPath) # file names in currect date directory
            for file in files:
                try:
                    if filePrefix in file:
                        filePath = os.path.join(currentPath, file)
                        filePaths.append(filePath)

                        fp = open(filePath, 'r')
                        datas_with_specific_date = json.load(fp)
                        # add observed data
                        for data in datas_with_specific_date:
                            data["ObservedDate"] = date.replace("-", "")
                            data["State"] = util.days_between(data["Date"], data["ObservedDate"]) - 1
                        data_decoded += datas_with_specific_date # do not use append function

                except:
                    print "Not a json file"
        except:
            print "Not a directory, MAC OS contains .DS_Store file."

    # filter the null entries
    data_decoded = filter(is_not_nullprice, data_decoded)

    return data_decoded


def load_data_with_daysBeforeTakeoff_and_sameFlightNum(days, filePrefix="BCN_BUD", dataset="large data set"):
    """
    Load data with same flight number and the same days before takeoff.
    i.e. same equivalence class
    But in out dataset, one route means one flight number.
    :param days: the days before takeoff
    :param filePrefix: choose which route
    :param dataset: choose from wchich dataset
    :return: data with same flight number and the same days before takeoff
    """
    datas = load_data_with_prefix_and_dataset(filePrefix, dataset)
    output = [data for data in datas if util.days_between(data["ObservedDate"], data["Date"]) == days]

    return output

def get_departure_len(filePrefix="BCN_BUD", dataset="large data set"):
    """
    So far, used in QLearning, return the total departure date length in the chosen dataset.
    """
    datas = load_data_with_prefix_and_dataset(filePrefix, dataset)

    # get different departure data in the same flight number,
    # to compute the Q Values for such (flight number, departure date) pair.
    departureDates = []
    [departureDates.append(data["Date"]) for data in datas]
    departureDates = util.remove_duplicates(departureDates)

    return len(departureDates)


def load_data_QLearning(departureIndex, filePrefix="BCN_BUD", dataset="large data set"):
    """
    Given the departureIndex, return the dataset with specific departure date in the chosen dataset.
    """
    datas = load_data_with_prefix_and_dataset(filePrefix, dataset)

    # get different departure data in the same flight number,
    # to compute the Q Values for such (flight number, departure date) pair.
    departureDates = []
    [departureDates.append(data["Date"]) for data in datas]
    departureDates = util.remove_duplicates(departureDates)

    # choose the departure date by departureIndex
    departureDate = departureDates[departureIndex]
    print "Evaluating departure date " + departureDate + "..."

    """
    # remove duplicate observedDate-departureDate pair
    observedDates = []
    [observedDates.append(data["ObservedDate"]) for data in datas if data["Date"]==departureDate]
    observedDates = util.remove_duplicates(observedDates)
    states = len(observedDates)
    #print states
    """


    specificDatas = []
    specificDatas = [data for data in datas if data["Date"]==departureDate]

    #states = []
    #[states.append(data["State"]) for data in specificDatas]
    #print max(states)

    return specificDatas

def getMinimumPrice(datas):
    """
    Given the dataset, return the minimum price in the dataset
    :param datas: input dataset(in QLearning, it should have same departure date)
    :return: minimum price in the dataset
    """
    minimumPrice = util.getPrice(datas[0]["MinimumPrice"]) # in our json data files, MinimumPrice means the price in that day
    for data in datas:
        price = util.getPrice(data["MinimumPrice"])
        minimumPrice = price if price<minimumPrice else minimumPrice
    minimumPrice = minimumPrice

    return minimumPrice

def getChosenPrice(state, datas):
    """
    Given the state, i.e. the days before departure, and the dataset, return the price
    :param state: the days before departure
    :param datas: input dataset(in QLearning, it should have same departure date)
    :return: the chosen price
    """
    for data in datas:
        if data["State"] == state:
            return util.getPrice(data["MinimumPrice"])


if __name__ == "__main__":
    datas = load_data_QLearning()
    print len(datas)












