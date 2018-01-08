import numpy as np
from numpy.linalg import inv
import scipy.optimize as optimization
from scipy.stats import norm
from CollectData28 import July_21_17
from CollectData28 import Sept_15_17
from CollectData28 import June_15_18
from CollectData28 import Jan_18_19
from CollectData28 import May_19_17
from CollectData28 import June_16_17
from CollectData28 import Oct_20_17
from CollectData28 import Jan_19_18
import timeit
start_BM_Ite = timeit.default_timer()



## Sifting
T = []
xdata = []
K = []
price_actual = []
weight = []

for data in range(8):
    if data == 0:
        for i in range(len(July_21_17)):
            if abs(July_21_17[i][1]-July_21_17[i][2]) <= 1 and July_21_17[i][3] >= 10:
                K.append(July_21_17[i][0])
                weight.append(1.0/abs(July_21_17[i][1]-July_21_17[i][2]))
                #weight.append(1.0)
                price_actual.append(0.5*(July_21_17[i][1] + July_21_17[i][2]))
                T.append(84)

    if data == 1:
        for i in range(len(Sept_15_17)):
            if abs(Sept_15_17[i][1]-Sept_15_17[i][2]) <= 1 and Sept_15_17[i][3] >= 10:
                K.append(Sept_15_17[i][0])
                weight.append(1.0 / abs(Sept_15_17[i][1]-Sept_15_17[i][2]))
                #weight.append(1.0)
                price_actual.append(0.5 * (Sept_15_17[i][1] + Sept_15_17[i][2]))
                T.append(140)

    if data == 2:
        for i in range(len(June_15_18)):
            if abs(June_15_18[i][1]-June_15_18[i][2]) <= 1 and June_15_18[i][3] >= 10:
                K.append(June_15_18[i][0])
                weight.append(1.0 / abs(June_15_18[i][1]-June_15_18[i][2]))
                #weight.append(1.0)
                price_actual.append(0.5 * (June_15_18[i][1] + June_15_18[i][2]))
                T.append(413)

    if data == 3:
        for i in range(len(Jan_18_19)):
            if abs(Jan_18_19[i][1]-Jan_18_19[i][2]) <= 1 and Jan_18_19[i][3] >= 10:
                K.append(Jan_18_19[i][0])
                weight.append(1.0 / abs(Jan_18_19[i][1]-Jan_18_19[i][2]))
                #weight.append(1.0)
                price_actual.append(0.5 * (Jan_18_19[i][1] + Jan_18_19[i][2]))
                T.append(630)

    if data == 4:
        for i in range(len(May_19_17)):
            if abs(May_19_17[i][1]-May_19_17[i][2]) <= 1 and May_19_17[i][3] >= 10:
                K.append(May_19_17[i][0])
                weight.append(1.0 / abs(May_19_17[i][1]-May_19_17[i][2]))
                #weight.append(1.0)
                price_actual.append(0.5 * (May_19_17[i][1] + May_19_17[i][2]))
                T.append(21)

    if data == 5:
        for i in range(len(June_16_17)):
            if abs(June_16_17[i][1]-June_16_17[i][2]) <= 1 and June_16_17[i][3] >= 10:
                K.append(June_16_17[i][0])
                weight.append(1.0 / abs(June_16_17[i][1]-June_16_17[i][2]))
                #weight.append(1.0)
                price_actual.append(0.5 * (June_16_17[i][1] + June_16_17[i][2]))
                T.append(49)

    if data == 6:
        for i in range(len(Oct_20_17)):
            if abs(Oct_20_17[i][1]-Oct_20_17[i][2]) <= 1 and Oct_20_17[i][3] >= 10:
                K.append(Oct_20_17[i][0])
                weight.append(1.0 / abs(Oct_20_17[i][1]-Oct_20_17[i][2]))
                #weight.append(1.0)
                price_actual.append(0.5 * (Oct_20_17[i][1] + Oct_20_17[i][2]))
                T.append(175)

    if data == 7:
        for i in range(len(Jan_19_18)):
            if abs(Jan_19_18[i][1]-Jan_19_18[i][2]) <= 1 and Jan_19_18[i][3] >= 10:
                K.append(Jan_19_18[i][0])
                weight.append(1.0 / abs(Jan_19_18[i][1]-Jan_19_18[i][2]))
                #weight.append(1.0)
                price_actual.append(0.5 * (Jan_19_18[i][1] + Jan_19_18[i][2]))
                T.append(266)

initial_parameters = np.array([0.8, -0.5])
K = np.asarray(K)
T = np.asarray(T)
price_actual = np.asarray(price_actual)
weight = np.asarray(weight)
xdata.append(T)
xdata.append(K)
xdata = np.asarray(xdata)

####
def BS_func(xdata, sigma):
    r = 0.0375
    S0 = 160.29
    K = xdata[1]
    T = xdata[0]*1.0/360
    d1 = (np.log(S0 * 1.0 / K) + (r + 0.5 * sigma ** 2) * T) * 1.0 / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C_model = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    #print(C_model)
    return C_model

####
def optionPrice1(xdata, sigma, betaPara):
    s = 160.29  ## The price of Apple Stock at time: 3:00 pm, date: 04/28/2017
    r = 0.0375  ## Yield at 04/28/2017
    T = xdata[0] / 360
    K = xdata[1]
    n = 100
    m = 1000
    s_max = 300
    ds = s_max * 1.0 / m
    position = round(s / ds)

    ##Construct A_star which is a m*m matrix
    A = np.zeros((m, m))
    si = np.linspace(1, m, m) * ds

    for i in range(1, len(A) - 1):
        A[i][i - 1] = 0.5 * ((sigma ** 2) * (si[i] ** (2 * betaPara + 2)) / (ds ** 2) - (r * si[i]) / ds)
        A[i][i] = -1.0 * (sigma ** 2) * (si[i] ** (2 * betaPara + 2)) / (ds ** 2) - r
        A[i][i + 1] = 0.5 * ((sigma ** 2) * (si[i] ** (2 * betaPara + 2)) / (ds ** 2) + (r * si[i]) / ds)
    A[0][0] = -1.0 * (sigma ** 2) * (si[0] ** (2 * betaPara + 2)) / (ds ** 2) - r
    A[0][1] = 0.5 * ((sigma ** 2) * (si[0] ** (2 * betaPara + 2)) / (ds ** 2) + (r * si[0]) / ds)
    A[-1][-2] = - r * s_max * 1.0 / ds
    A[-1][-1] = r * s_max * 1.0 / ds - r


    model_value = []
    for times in range(len(K)):
        dt = T[times] * 1.0 / n
        ## CrankNicolson Scheme
        A_star_1 = inv(np.eye(len(A)) - 0.5*dt*A)
        A_star_2 = np.eye(len(A)) + 0.5*dt*A
        A_star = np.dot(A_star_1, A_star_2)

        ## Implicit Scheme
        #A_star = inv(np.eye(len(A)) - dt*A)

        ## Explicit Scheme
        #A_star = np.eye(len(A)) + dt*A

        # Option price V
        V = np.zeros((n+1, m+1))

        # Set initial data
        for i in range(m + 1):
            V[0][i] = max(i * ds - K[times], 0)

        # Set interior data and higher boundary data
        for i in range(1, n + 1):
            V[i][1:] = np.dot(A_star, V[i - 1][1:])
        model_value.append(V[n][position])
    #print(model_value)
    model_value = np.asarray(model_value)
    return model_value


parameters1 = optimization.curve_fit(optionPrice1, xdata, price_actual, initial_parameters, 1.0/weight,
                                     bounds=([0., -1], [np.inf, 0.]))[0]
parameters2 = optimization.curve_fit(BS_func, xdata, price_actual, initial_parameters[0], 1.0/weight,  bounds=(0., np.inf))[0]
stop_BM_Ite = timeit.default_timer()
print("Calibrated by CEV model with initial sigma= "+str(initial_parameters[0])+", initial beta= "+str(initial_parameters[1])
      +":"+"\nThe result is: sigma= "+str(parameters1[0]) + " ,beta= "+str(parameters1[1]))
print("\n")
print("Calibrated by BS Formula with initial sigma= "+str(initial_parameters[0])+": "
      +"\nThe result is: sigma= "+str(parameters2))
print("\n")
print("Time consuming: "+str(stop_BM_Ite-start_BM_Ite)+" s")

#print(optionPrice1(xdata,0.2,0))
#print(BS_func(xdata, 0.2))
#print(K)





