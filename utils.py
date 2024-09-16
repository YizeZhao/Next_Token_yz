import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from collections import Counter
import cvxpy as cp


def SVM_thm_sol(id_ctx_dict, support_set_sampled, v_nt):
    print("computing Lmm in SVM the sol")
    m = len(id_ctx_dict.keys())
    L_mm = np.zeros((m, v_nt))
    for k, id_ctx in id_ctx_dict.items():
        j = id_ctx['id']
        support = support_set_sampled[k]
        s = len(support)
        L_mm[j] = np.ones((1, v_nt)) * -s/v_nt
        L_mm[j][support] = (v_nt-s)/v_nt

    return L_mm

def SVM_thm_sol2(S):
    m, V = S.shape
    L_claim = np.zeros((V, m))
    for j in range(m):
        for z in range(V):
            k = np.sum(S[j, :])
            if S[j, z] == 1:
                L_claim[z, j] = (V - k) / V
            else:
                L_claim[z, j] = -k / V

    return L_claim.T

def get_support_gram(id_ctx_dict, support_set_sampled, v_nt):
    m = len(id_ctx_dict.keys())
    L_s = np.zeros((m, v_nt))
    for k, id_ctx in id_ctx_dict.items():
        j = id_ctx['id']
        support = support_set_sampled[k]
        k = len(support)
        L_s[j] = np.zeros((1, v_nt))
        L_s[j][support] = 1

    # G_Ls = L_s.T @ L_s
    G_Ls = L_s @ L_s.T

    return G_Ls

def get_s(id_ctx_dict, support_set_sampled, v_nt):
    m = len(id_ctx_dict.keys())
    L_s = np.zeros((m, v_nt))
    for k, id_ctx in id_ctx_dict.items():
        j = id_ctx['id']
        support = support_set_sampled[k]
        k = len(support)
        L_s[j][support] = 1

    # G_Ls = L_s.T @ L_s
    # G_Ls = L_s @ L_s.T
    return L_s

def get_p(id_ctx_dict, support_set_sampled, support_set_pr,  v_nt):
    m = len(id_ctx_dict.keys())
    P_s = np.zeros((m, v_nt))
    for k, id_ctx in id_ctx_dict.items():
        j = id_ctx['id']
        support = support_set_sampled[k]
        P_s[j][support] = support_set_pr[k]

    return P_s

def proj_f_w(Smat, W):
    W_shape = W.shape
    U, S, Vh = np.linalg.svd(Smat, full_matrices=False)
    S_inrank = np.where(S > 1e-4)[0]
    W_proj = U[:,S_inrank] @ U[:,S_inrank].T @ W.flatten('F').reshape([-1,1])
    W_proj = W_proj.reshape(W_shape, order='F')

    return W_proj

# def compute_norm_corr(a_list, b):
#     # assume that a is normed
#     a_list_normed = np.divide(a_list,np.linalg.norm(a_list, axis=(1,2))[:, np.newaxis, np.newaxis])
#     # b = np.divide(b,np.linalg.norm(b, axis=(1,2))[:, np.newaxis, np.newaxis])
#     a_diff = np.linalg.norm(a_list_normed - b, axis=(1,2))
#     # a_diff = []
#     # for i in range(len(a_list)):
#     #     a_t = a_list[i]
#     #     a_diff.append(norm(a_t / norm(a_t, 'fro') - a / norm(a, 'fro')))
#     #     # a_diff.append(norm(a_t  - a )/ norm(a, 'fro'))
#
#     return a_diff

def compute_norm_corr(a, b, order='fro'):
    # assume that a is normed
    print(a.shape, b.shape)
    if len(a.shape) == 2:
        a_normed = np.divide(a,np.linalg.norm(a, ord=order))
    elif len(a.shape) == 3:
        a_normed = np.divide(a, np.linalg.norm(a, axis=(1, 2), ord=order)[:, np.newaxis, np.newaxis])

    if len(b.shape) == 2:
        b_normed = np.divide(b,np.linalg.norm(b, ord=order))
    elif len(b.shape) == 3:
        b_normed = np.divide(b, np.linalg.norm(b, axis=(1, 2), ord=order)[:, np.newaxis, np.newaxis])

    if len(a.shape) == 2 and len(b.shape) == 2:
        diff = np.linalg.norm(a_normed - b_normed)
    else:
        diff = np.linalg.norm(a_normed - b_normed, axis=(1, 2))

    return diff

def find_Smat(uniq_embeds_dict, support_set_sampled, s_len, v_nt, test_smat=False):
    '''
    1 consturct S_mat
    2. SVM getU
    3. compute projected W and reshape
    :return:
    '''
    print("finding Smat")

    Smat = []  # measurement vectors (e_z-e_z')^Th_j
    test_h=None

    for k, h in uniq_embeds_dict.items():
        support = support_set_sampled[k]
        h = h.reshape(1, -1)
        assert len(support) == s_len
        for z in range(s_len):
            e_z = np.zeros((v_nt))
            e_z[support[z]] = 1
            for z_ in range(z+1, s_len):
                e_z_ = np.zeros((v_nt))
                e_z_[support[z_]] = 1
                smat_temp = (e_z - e_z_).reshape(-1,1) @ h
                if test_h is None:
                    test_h = smat_temp.copy()
                    test_shape = smat_temp.shape
                smat_temp = smat_temp.flatten('F')
                Smat.append(smat_temp)
    Smat = np.array(Smat).T

    if test_smat:
        U, S, Vh = np.linalg.svd(Smat, full_matrices=False)
        test_h_proj = U @ U.T @ test_h.flatten('F')
        test_h_proj = test_h_proj.reshape(test_shape, order='F')
        test_diff = norm(test_h_proj - test_h)
        print(test_diff)

    return Smat


def matrix_sqrt(A):
    # Perform Singular Value Decomposition
    U, S, VT = np.linalg.svd(A)

    # Take the square root of the singular values
    S_sqrt = np.sqrt(S)
    S_diag = np.diag(S)

    # Construct the diagonal matrix of square roots of singular values
    S_sqrt_diag = np.diag(S_sqrt)

    # Adjust the size of S_sqrt_diag to match U and VT
    S_sqrt_diag = np.dot(U, np.dot(S_sqrt_diag, VT[:len(S_sqrt), :]))
    # S_sqrt_diag = np.dot(U, np.dot(S_diag, VT[:len(S_sqrt), :]))


    # Compute the square root of the original matrix
    # A_sqrt = np.dot(U, np.dot(S_sqrt_diag, VT))

    return S_sqrt_diag


def compute_grams(L_mm, v_nt=0):
    u, s, vh = np.linalg.svd(L_mm, full_matrices=False)

    #print(f"Singular values of Lmm: {s}")
    rank = len(np.where(s > 1e-6)[0])
    print(f"Rank of Lmm: {np.linalg.matrix_rank(L_mm)}")
    print(f"rank of Lmm by singular values: {len(np.where(s > 1e-6)[0])}")

    # plt.plot(s)
    # plt.show()
    # s = s[:v_nt-1]
    # u = u[:,:v_nt-1]
    # vh = vh[:v_nt-1,:]
    s = s[:rank]
    u = u[:,:rank]
    vh = vh[:rank,:]

    WG_mm = vh.T @ np.diag(s) @ vh
    W_mm = vh.T @ np.diag(np.sqrt(s))
    HG_mm = u @ np.diag(s) @ u.T
    H_mm = u @ np.diag(np.sqrt(s))


    WG_mm_norm = WG_mm / norm(WG_mm,'fro')
    HG_mm_norm = HG_mm / norm(HG_mm,'fro')
    # print(WG_mm.shape)
    # print(HG_mm.shape)
    return WG_mm_norm, HG_mm_norm, W_mm, H_mm, u, vh, WG_mm, HG_mm

def project_L(L, u, vh):
    pass

def project_W2mm(gw, vh):
    # GW_proj = []
    # for gw in GW_list:
    #     # gw_proj = u @ u.T @ gw.flatten('F').reshape([-1,1])
    #     # W_proj = W_proj.reshape(gw.shape, order='F')
    #     # print(norm(W_proj - gw))
    gw_proj = vh.T @ vh @ gw @ vh.T @ vh
    #     GW_proj.append(gw_proj)
    # return GW_proj
    return gw_proj


def compute_svm(init_from, uniq_embeds_dict, support_set_sampled, support_set_pr, d_decode, v_nt):
    if init_from == "mlp":
        svm = MultiLabelSVM(
            emb_dict=uniq_embeds_dict, support_dict=support_set_sampled, prob_dict=support_set_pr, d=d_decode,
            v_nt=v_nt)
    elif init_from == "tfm":
        svm = MultiLabelSVM(
            emb_dict=uniq_embeds_dict, support_dict=support_set_sampled, prob_dict=support_set_pr, d=d_decode,
            v_nt=v_nt)
    elif init_from == "ufm":
        svm = MultiLabelSVM(
            emb_dict=uniq_embeds_dict, support_dict=support_set_sampled, prob_dict=support_set_pr, d=d_decode,
            v_nt=v_nt)
        # print(v_nt)

    else:
        raise ValueError("model not defined")

    w_fin = svm.find_Wfin()
    w_star = svm.find_WStar()

    #
    # if init_from == "ufm":
    #     return w_fin_normed, w_star_normed, l_mm_normed

    return w_fin, w_star

def clean_dict(ctx_id_dict, emb_dict, support_sampled_dict, support_pr_dict, support_set_repeats):
    all_dict = {}
    for k, v in ctx_id_dict.items():
        all_dict[k] = {}
        all_dict[k]["id"] = ctx_id_dict[k]["id"]
        all_dict[k]["value"] = ctx_id_dict[k]["value"]
        # all_dict[k]["emb"] = emb_dict[k]
        all_dict[k]["support"] = support_sampled_dict[k]
        all_dict[k]["prob"] = support_pr_dict[k]
        all_dict[k]["repeats"] = support_set_repeats[k]
    return all_dict

def pairwise_cossim(A):
    # compute pairwise cos of rows in a matrix
    sim = 1 - cdist(A, A, metric='cosine')
    return sim


def plot_all_dict(sp, all_dict):
    nrows, ncols = 4, 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), dpi=100)
    keys = list(all_dict.keys())
    cnt = 0
    for i in range(nrows):
        for j in range(ncols):
            s, reps = [], []
            k = keys[cnt]
            ax = axs[i][j]
            ctx = [sp.IdToPiece(int(id_)) for id_ in all_dict[k]["value"]]
            # support = [sp.IdToPiece(int(id_)) for id_ in all_dict[k]["support"]]
            repeats = all_dict[k]["repeats"]
            repeats_cnt = Counter(repeats)
            for t_, freq_ in repeats_cnt.items():
                s.append(t_)
                reps.append(freq_)
            support = [sp.IdToPiece(int(id_)) for id_ in s]
            ax.bar(support, reps)
            ax.set_title(" ".join(ctx))
            ax.set_ylim([0, 5])

            cnt += 1

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f'./figures/dataset.png')

def SVM_cvxpy_sol(S):
    print("Computing Lmm with cvxpy")
    m, V = S.shape
    L = cp.Variable((m,V))
    cost = cp.atoms.norm(L, 'nuc')
    constraints = []
    for j in range(m):
        z = np.argwhere(S[j,:]==1)[0,0]
        # for z in supports:
        for z_ in range(V):
            if z == z_:
                continue
            if S[j,z_] == 1:
                constraints.append((L[j,z] - L[j,z_]) == 0)
            elif S[j,z_] == 0:
                constraints.append((L[j,z] - L[j,z_]) >= 1)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(verbose=True, eps_abs=1e-10, )
    return L.value

def compute_L_fin(P, S):
    assert P.shape == S.shape
    m, V = S.shape
    L_star = np.zeros((m, V))
    for j in range(m):
        anchor = np.argwhere(S[j, :] == 1)[0, 0]
        idx_list = list(np.argwhere(S[j, :] == 1)[1:, 0])
        E_j = np.zeros((len(idx_list), V))
        a_j = np.zeros((len(idx_list), 1))
        for i, id in enumerate(idx_list):
            E_j[i, anchor] = 1
            E_j[i, id] = -1
            a_j[i] = np.log(P[j, anchor] / P[j, id])

        L_star[j, :] = np.squeeze(E_j.T @ np.linalg.inv(E_j @ E_j.T) @ a_j)

    return L_star

def project_Lt(P, S, L_list, L_star):
    assert P.shape == S.shape
    m, V = S.shape
    L_proj_L_fin_diff = []
    for L_t in L_list:
        L_t_proj = np.zeros((m, V))

        for j in range(m):
            anchor = np.argwhere(S[j, :] == 1)[0, 0]
            idx_list = list(np.argwhere(S[j, :] == 1)[1:, 0])

            E_j = np.zeros((len(idx_list), V))
            a_j = np.zeros((len(idx_list), 1))
            for i, id in enumerate(idx_list):
                E_j[i, anchor] = 1
                E_j[i, id] = -1
                a_j[i] = np.log(P[j, anchor] / P[j, id])

            L_t_proj[j, :] = E_j.T @ np.linalg.inv(E_j @ E_j.T) @ E_j @ L_t[j, :]

        L_proj_L_fin_diff.append(norm(L_t_proj - L_star))
    return L_proj_L_fin_diff

# def check_lmm_constraint(S, L_to_check):
#     print("Checking constraints for Lmm")
#     m, V= S.shape
#     L = L_to_check.T
#     largest_tol = 0
#     sat = True
#     for j in range(m):
#         anchor = np.argwhere(S[j,:]==1)[0,0]
#         for z in range(V):
#           if z != anchor:
#             if S[j,z] == 1:
#               if (np.abs(L[anchor,j] - L[z,j]) >= 1e-7):
#                   tol = np.abs(L[anchor,j] - L[z,j])
#                   print(f"=0 Constraint violated for j={j} z={z} with value L[v,j] - L[z,j] =  {(L[anchor,j] - L[z,j])}")
#                   if tol > largest_tol:
#                       largest_tol = tol
#                   sat = False
#             elif S[j,z] == 0:
#               if (L[anchor,j] - L[z,j] < 1 - 1e-7):
#                   tol = 1 - (L[anchor,j] - L[z,j])
#                   if tol > largest_tol:
#                       largest_tol = tol
#                   print(f">= 1 Constraint violated for j={j} z={z} with value L[v,j] - L[z,j] - 1 =  {(L[anchor,j] - L[z,j]) - 1}")
#                   sat = False
#
#     print(f"Largest violation: {largest_tol}")
#     return sat

