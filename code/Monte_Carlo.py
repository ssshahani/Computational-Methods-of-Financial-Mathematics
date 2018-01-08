import numpy as np
from prettytable import PrettyTable

def CEV(s,K,B,type):
    sigma = 0.105516623093
    beta = -4.47444*np.power(10, -14)
    r = 0.0375
    n = 5000
    T = 1
    dt = 1.0*T/ 360
    payoff_array = []

    if type == "up_and_out":
        for j in range(n):
            hit_barrier = False
            St = []
            St.append(s)
            ## Normal variable
            Z = np.random.standard_normal(size=360-1)
            for i in range(1, 360):
                St_i = St[i-1] + r*St[i-1]*dt + sigma*np.sqrt(dt)*Z[i-1]*(St[i-1]**(beta + 1)) \
                            + 0.5*(sigma**2)*(beta + 1)*(dt*(Z[i-1]**2) - dt)*(St[i-1]**(2*beta+1))
                if (i+1)%30==0:  # Monthly Monitor
                    if St_i > B: ## Up and out option
                        hit_barrier = True
                        break  # If St goes above the barrier, stop by setting time
                St.append(St_i)
            if hit_barrier == False:
                payoff = max(St[-1] - K, 0)*np.exp(-r*T)
                payoff_array.append(payoff)
            else:
                payoff_array.append(0)
        price = sum(payoff_array) * 1.0 / n

    elif type == "up_and_in":
        for j in range(n):
            hit_barrier = True
            St = []
            St.append(s)
            ## Normal variable
            Z = np.random.standard_normal(size=360-1)
            for i in range(1, 360):
                St_i = St[i-1] + r*St[i-1]*dt + sigma*np.sqrt(dt)*Z[i-1]*(St[i-1]**(beta + 1)) \
                            + 0.5*(sigma**2)*(beta + 1)*(dt*(Z[i-1]**2) - dt)*(St[i-1]**(2*beta+1))
                if (i+1)%30==0:  # Monthly Monitor
                    if St_i >= B: ## Up and in option
                        hit_barrier = False
                St.append(St_i)
            if hit_barrier == False:
                payoff = max(St[-1] - K, 0)*np.exp(-r*T)
                payoff_array.append(payoff)
            else:
                payoff_array.append(0)
        price = sum(payoff_array) * 1.0 / n

    elif type == "down_and_out":
        for j in range(n):
            hit_barrier = False
            St = []
            St.append(s)
            ## Normal variable
            Z = np.random.standard_normal(size=360-1)
            for i in range(1, 360):
                St_i = St[i-1] + r*St[i-1]*dt + sigma*np.sqrt(dt)*Z[i-1]*(St[i-1]**(beta + 1)) \
                            + 0.5*(sigma**2)*(beta + 1)*(dt*(Z[i-1]**2) - dt)*(St[i-1]**(2*beta+1))
                if (i+1)%30==0:  # Monthly Monitor
                    if St_i < B: ## Down and out option
                        hit_barrier = True
                        break  # If St goes above the barrier, stop by setting time
                St.append(St_i)
            if hit_barrier == False:
                payoff = max(St[-1] - K, 0)*np.exp(-r*T)
                payoff_array.append(payoff)
            else:
                payoff_array.append(0)
        price = sum(payoff_array) * 1.0 / n

    elif type == "down_and_in":
        for j in range(n):
            hit_barrier = True
            St = []
            St.append(s)
            ## Normal variable
            Z = np.random.standard_normal(size=360-1)
            for i in range(1, 360):
                St_i = St[i-1] + r*St[i-1]*dt + sigma*np.sqrt(dt)*Z[i-1]*(St[i-1]**(beta + 1)) \
                            + 0.5*(sigma**2)*(beta + 1)*(dt*(Z[i-1]**2) - dt)*(St[i-1]**(2*beta+1))
                if (i+1)%30==0:  # Monthly Monitor
                    if St_i <= B: ## Down and in option
                        hit_barrier = False
                St.append(St_i)
            if hit_barrier == False:
                payoff = max(St[-1] - K, 0)*np.exp(-r*T)
                payoff_array.append(payoff)
            else:
                payoff_array.append(0)
        price = sum(payoff_array) * 1.0 / n
    return price

def optionType(type):
    K = np.array([150,155,160,165,170,175,180])
    B = np.array([140,150,160,170,180])
    print("Type is: "+ type)
    x = PrettyTable(["Value", "B = 140", "B = 150", "B = 160", "B = 170", "B = 180"])
    x.align["Value"] = "l"
    x.padding_width = 1
    for i in range(len(K)):
        x.add_row(["K = " + str(K[i]), CEV(160.3,K[i],B[0],type), CEV(160.3,K[i],B[1],type),
                   CEV(160.3, K[i], B[2], type), CEV(160.3,K[i],B[3],type), CEV(160.3,K[i],B[4],type)])
    print(x)
optionType("down_and_out")

