#!/usr/bin/env python3


import numpy as np


# def init_cars(M):
#     return np.arange(0,M, dtype='int')

def init_cars(M,N):
    cars = np.zeros(N, dtype=bool)
    cars[:M] = True
    return cars

def valid_moves(cars):
    result = np.zeros_like(cars, dtype=bool)
    result[:-1] = cars[:-1] & ~cars[1:]
    result[-1] = cars[-1] & ~cars[0]
    return np.where(result)[0]
    
def move_cars(i, cars):
    cars2 = np.copy(cars)
    cars2[i]=False
    cars2[(i+1)%len(cars)] = True
    return cars2

def propogate(cars):
    pos_moves = valid_moves(cars)
    result = []
    for move in pos_moves:
        c2 = move_cars(move,cars)
        result.append(c2)
    return result

def run_sim(cars_dict, T):
    if T == 0:
        return cars_dict
    new_cars_dict = {}
    # For each probable state, we compute the possible moves and
    # propagate the probabilities of ending up each state at next
    # step. We repeat this process until we iterated as many times as
    # needed. We save a lot of computation time by realizing that it
    # does not matter how we got to a state, so each outcome state is
    # simply the sum of probabilities of each initial state times the
    # probability of the initial state propagating to that final
    # state.
    for _, cars_set in cars_dict.items():
        cars = cars_set[0]
        weight = cars_set[1]
        propogated_cars = propogate(cars)
        for new_cars in  propogated_cars:
            key = new_cars.tobytes()
            elem = new_cars_dict.setdefault(key, [new_cars, 0.0])
            elem[1]+=weight/len(propogated_cars)
    return run_sim(new_cars_dict, T-1)

def compute_quantities(cars):
    pos = np.arange(len(cars))[cars]
    return np.average(pos), np.std(pos)


def get_answers(N,M,T):
    # The state of the cars is saved as a boolean array with one entry
    # per node on the graph. Individual car identities do not matter.
    cars = init_cars(M, N)
    # I place the initial state of the cars into a dictionary that contains
    # the set of possible states and the probability of each state. Currently
    # there is only one possible state: the initial state with p=1. 
    cars_dict= {cars.tobytes(): [cars,1.0]}
    final_dist = run_sim(cars_dict,50)
    averages = np.zeros(len(final_dist))
    stds     = np.zeros(len(final_dist))
    weights  = np.zeros(len(final_dist))
    for i,(_, dist) in enumerate(final_dist.items()):
        avg,std= compute_quantities(dist[0])
        averages[i] = avg
        stds[i]     = std
        weights[i]  = dist[1]
    print("\n\nN={} M={} T={}".format(N,M,T))
    print("\t<average>: ", np.average(averages, weights = weights))
    print("\tstd of average: ", np.average((averages-np.average(averages, weights = weights))**2, weights=weights))
    print("\t<std>: ", np.average(stds, weights = weights))
    print("\tstd of average: ", np.average((stds-np.average(stds, weights = weights))**2, weights=weights))


if __name__ == "__main__":
    get_answers(10,5,20)
    get_answers(25,10,50)
