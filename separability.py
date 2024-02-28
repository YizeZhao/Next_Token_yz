'''
find 9: find a W that satisfie 9
find 10: find a W that satisfied 10
check 9: given a W see if 9 satisfies
check 10: given a W see if 10 satisfies
'''
import numpy as np
import cvxpy as cp
import random
import matplotlib.pyplot as plt


def conditional_entropy(p_s, s_sets):
    for j in range(p_s.shape(0)):
        non_zeros = np.where(p_s[j] != 0)
        #cond_entropu_j =
        #TODO: finish


def find_10(h_s, s_s):
    W_star_cvx = cp.Variable((v,d))
    obj_10 = cp.Minimize(0)
    constraint_list = []
    for j in range(h_s.shape[0]):
        for z in s_s[j]:
            for z_ in range(v):
                if z_ == z:
                    continue
                elif z_ in s_s[j]:
                    constraint_list.append((W_star_cvx[z] - W_star_cvx[z_]).T@(h_s[j]) == 0)
                else:
                    constraint_list.append((W_star_cvx[z] - W_star_cvx[z_]).T@(h_s[j]) >= 1)

    const_10 = constraint_list
    prob_10 = cp.Problem(obj_10, const_10)

    result = prob_10.solve(cp.SCS, verbose=True)
    W_star = W_star_cvx.value
    #W_star = np.random.normal(size=(v,d))


    if np.isnan(result):
        print('no solution found for W_star')

    # check_list = []
    # for j in range(h_s.shape[0]):
    #     for z in s_s[j]:
    #         for z_ in range(v):
    #             if z_ == z:
    #                 continue
    #             elif z_ in s_s[j]:
    #                 check_list.append(abs((W_star[z] - W_star[z_]).T@(h_s[j])) <= 1e-10)
    #             else:
    #                 check_list.append((W_star[z] - W_star[z_]).T@(h_s[j]) > 1)

    return W_star


def find_9(h_s, s_s, p_s):
    W_fin_cvx = cp.Variable((v,d))
    obj_9 = cp.Minimize(0)
    constraint_list = []
    for j in range(h_s.shape[0]):
        for z in s_s[j]:
            for z_ in range(v):
                if z_ == z:
                    continue
                elif z_ in s_s[j]:
                    constraint_list.append((W_fin_cvx[z] - W_fin_cvx[z_]).T@(h_s[j]) == cp.log(p_s[j][z]) - cp.log(p_s[j][z_]))

    const_9 = constraint_list
    prob_9 = cp.Problem(obj_9, const_9)

    result = prob_9.solve( verbose=True)
    W_fin = W_fin_cvx.value
    #W_star = np.random.normal(size=(v,d))


    if np.isnan(result):
        print('no solution found for W_fin')

    # check_list = []
    # for j in range(h_s.shape[0]):
    #     for z in s_s[j]:
    #         for z_ in range(v):
    #             if z_ == z:
    #                 continue
    #             elif z_ in s_s[j]:
    #                 check_list.append(abs((W_star[z] - W_star[z_]).T@(h_s[j])) <= 1e-10)
    #             else:
    #                 check_list.append((W_star[z] - W_star[z_]).T@(h_s[j]) > 1)

    return W_fin

def generate_data(m, d, v, b):

    h_s = np.random.normal(0, 1, size=(m, d))
    s_bags = np.zeros((m,b)).astype(int)
    p_s = np.zeros((m, v))
    s_sets = {}
    for j in range(m):
        support_j = np.random.choice(np.arange(0,v, dtype=int), size=(b), replace=True)
        s_bags[j] = support_j
        s_set, cnt = np.unique(support_j, return_counts=True)
        s_sets[j] = s_set
        for support in range(len(s_set)):
            p_s[j][s_set[support]] = cnt[support]/len(support_j)

    p_s = p_s/p_s.sum(axis=1,keepdims=1)
    return h_s, s_sets, p_s



if __name__ == '__main__':
    m = 20
    d = 20
    v = 50
    b = 10


    h_s, s_s, p_s = generate_data(m,d,v,b)
    W_star = find_10(h_s, s_s)
    # W_fin = find_9(h_s, s_s, p_s)
