import requests
import pandas
import scipy
import numpy
import sys
 
 
TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"
 
 
def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.
    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
 
    # YOUR IMPLEMENTATION HERE
    data_str = response.text.split()
    areas = numpy.array(list(map(float, data_str[0].split(',')[1:])))
    prices = numpy.array(list(map(float, data_str[1].split(',')[1:])))
 
    area_min, area_max = numpy.min(areas), numpy.max(areas)
    prices_min, prices_max = numpy.min(prices), numpy.max(prices)
 
    areas = (areas-area_min)/(area_max-area_min)
    prices = (prices-prices_min)/(prices_max-prices_min)
 
 
    w0 = numpy.random.random()
    w1 = numpy.random.random()
    w2 = numpy.random.random()
 
    b = 1.0
    lrate = 0.005
 
    EPoc = 500
    for epoch in (range(EPoc)):
 
        loss = (w0 * areas + w1 * (areas)*(0.6) + w2 * (areas)*2 - prices)
 
 
 
 
        gradw_0 = numpy.mean(2*loss*areas)
        gradw_1 = numpy.mean(2*loss*(areas**(0.5)))
        gradw_2 = numpy.mean(2*loss*(areas**2))
        gradb = numpy.mean(2*loss)
 
        w0 = w0 - lrate*gradw_0
        w1 = w1 - lrate*gradw_1
        w2 = w2 - lrate*gradw_2
        b = b - lrate*gradb
 
    area = (area-area_min)/(area_max-area_min)
    price = (w0 * area + w1 * (area)*(0.6) + w2 * (area)*2)
    price = price*(prices_max-prices_min) + prices_min
    return price
 
 
 
 
 
if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    print(rmse)
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
