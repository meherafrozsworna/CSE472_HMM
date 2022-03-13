import math
from scipy.stats import norm
import numpy as np

def readdata():
    global data
    global number_of_states
    global transition_matrix
    global mean
    global sigma_std

    # reading data
    data_file_name = './input/data.txt'
    data = []
    file_data = open(data_file_name)
    for line in file_data:
        data.append(float(line))

    file_data.close()

    # reading parameters

    parameter_file_name = './input/parameters.txt.txt'
    file_para = open(parameter_file_name, 'r')
    lines_list = file_para.readlines()
    number_of_states = int(lines_list[0])
    print(number_of_states)
    transition_matrix = []
    for i in range(1,number_of_states+1):
        x = lines_list[i].split()
        x = list(map(float, x))
        transition_matrix.append(x)
    print(transition_matrix)
    mean = lines_list[number_of_states+1].split()
    mean = list(map(float, mean))
    print(mean)
    variance = lines_list[number_of_states + 2].split()
    variance = list(map(float, variance))
    sigma_std = []
    for x in variance:
        sigma_std.append(math.sqrt(x))
    print(sigma_std)
    file_para.close()


readdata()

# global data
#     global number_of_states
#     global transition_matrix
#     global mean
#     global sigma_std
def emission_prob_calculate():
    emission_prob = []
    for i in range(number_of_states):
        state_em = []
        for d in data:
            # print(norm.pdf(d, mean[i], sigma_std[i]))
            # emission = (1/(sigma_std[i])*math.sqrt(2*math.pi)) * ( (math.e) ** (-0.5 * ((d - mean[i])/sigma_std[i]) ** 2) )
            emission = norm.pdf(d, mean[i], sigma_std[i])
            state_em.append(emission)
        emission_prob.append(state_em)
    return emission_prob


def stationary_distribution_calculate():
    arr = np.array(transition_matrix)

    #We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
    evals, evecs = np.linalg.eig(arr.T)
    evec1 = evecs[:,np.isclose(evals, 1)]

    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    print(evec1)
    evec1 = evec1[:,0]
    print("AAAAAAA")
    print(evec1)
    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary = stationary.real
    print("stationary : ",stationary)
    return stationary


def viterbi_table_calculation():
    emit_p = emission_prob_calculate()
    start_p = stationary_distribution_calculate()
    viterbi_table = [{}]
    for state in range(number_of_states):
        viterbi_table[0][state] = {
            "prob": math.log(start_p[state]) + math.log(emit_p[state][0]),
            "prev": None
        }

    # Run Viterbi when t > 0
    for i in range(1, len(data)):
        viterbi_table.append({})
        for state in range(number_of_states):
            max_transition_probability = viterbi_table[i - 1][0]["prob"] + math.log(transition_matrix[0][state])

            prev_state_selected = 0
            for prev_state in range(1, number_of_states):
                transition_probability = viterbi_table[i - 1][prev_state]["prob"] + math.log(transition_matrix[prev_state][state])
                if transition_probability > max_transition_probability:
                    max_transition_probability = transition_probability
                    prev_state_selected = prev_state

            # print(" max_transition_probability : ", max_transition_probability)
            max_prob = max_transition_probability + math.log(emit_p[state][i])
            viterbi_table[i][state] = {"prob": max_prob, "prev": prev_state_selected}

    return viterbi_table

def estimated_states(viterbi_table):
    # backtrack
    max_probability = - math.inf
    hidden_states = []
    best_st = None
    # Get most probable state and its backtrack

    for i in range(len(viterbi_table[-1])):
        state = viterbi_table[-1][i]["prev"]
        prob = viterbi_table[-1][i]["prob"]
        if prob > max_probability:
            max_probability = prob
            best_st = state


    hidden_states.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(viterbi_table) - 2, -1, -1):
        hidden_states.insert(0, viterbi_table[t + 1][previous]["prev"])
        # print(hidden_states)
        previous = viterbi_table[t + 1][previous]["prev"]

    return hidden_states

def viterbi():
    viterbi_table = viterbi_table_calculation()
    hidden_states = estimated_states(viterbi_table)
    f = open("states_Viterbi_wo_learning.txt", "w")
    for i in range(len(hidden_states)):
        if hidden_states[i] == 0:
            f.write("\"El Nino\"\n")
        else :
            f.write("\"La Nina\"\n")
    f.close()


viterbi()