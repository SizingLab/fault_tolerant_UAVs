"""
Controllability assessment and fault-tolerant sizing of UAVs.
"""

import numpy as np
from numpy import radians as rad
import itertools
from scipy.optimize import minimize, NonlinearConstraint
from scipy.linalg import null_space
from numpy.linalg import matrix_rank, norm


def xacai(Bf_l, Bf_nl, delta_min_l, delta_max_l, delta_min_nl, delta_max_nl, u0):
    """
    Computes the extended available control authority index.

    :param Bf_l: linear part of control effectiveness matrix
    :param Bf_nl: non-linear part of control effectiveness matrix
    :param delta_min_l: min effectors vector (linear part)
    :param delta_max_l: max effectors vector (linear part)
    :param delta_min_nl: min effectors vector (non-linear part)
    :param delta_max_nl: max effectors vector (non-linear part)
    :param u0: reference effort to provide

    :return rho: extended available control authority index
    """

    # control effectiveness matrix
    Bf = Bf_l + Bf_nl

    # center of the control space
    # expressed in the effectors frame
    delta_c_l = (delta_min_l + delta_max_l) / 2  # linear part
    delta_c_nl = (delta_min_nl + delta_max_nl) / 2  # non-linear part
    # expressed in the system frame ([X Y Z L M N])
    Uc = np.matmul(Bf_l, delta_c_l) + np.matmul(Bf_nl, delta_c_nl)
    # vector of max control distances to center point (in effectors frame)
    z_l = delta_max_l - delta_c_l  # linear part
    z_nl = delta_max_nl - delta_c_nl  # non-linear part

    # dimensions
    sz = Bf.shape
    n = sz[0]  # control space dimension ([T L M N] --> n = 4)
    m = sz[1]  # number of effectors (rotors)

    # M : list of rotors indices
    M = np.arange(0, m)
    # S1 : matrix of all possible combination of (n-1) rotors
    S1 = [list(l) for l in list(itertools.combinations(M, n - 1))]
    # sm : number of combinations of (n-1) rotors
    sm = len(S1)

    # Compute the distances from u0 to the boundary of Omega for each projection
    d_min = np.zeros(sm)  # vector of minimum distances from G to boundary of Omega for each projection
    for j in range(0, sm):
        # STEP 1: Define vector orth. to hypersegment col-space
        choose = S1[j]  # j-th combination of (n-1) effectors
        # B1: reduced control effectiveness matrix with (n-1) effectors (hyperspace row-space)
        B1 = Bf[:, choose]
        # find a n-dimensional vector orthonormal to the hyperspace defined by the (n-1) isolated effectors
        xsi = null_space(B1.transpose())  # orthonormal vector(s) of the null space of B1 (i.e., eigenvector(s))
        xsi = xsi[:, 0]  # take first column (eigenvector) in case of multiple solution

        # STEP 2: Truncate control effectiveness matrix and control vectors
        # B2: reduced control effectiveness matrix with the remaining (m-(n-1)) effectors
        B2_l_i = np.delete(Bf_l, choose, 1)
        B2_nl_i = np.delete(Bf_nl, choose, 1)
        # z: vector of max control distances to center point (only for (m-(n-1)) remaining effectors).
        z_l_i = np.delete(z_l, choose)
        z_nl_i = np.delete(z_nl, choose)
        # E: projection of remaining (m-(n-1)) effectors on the normal vector of the hyperplane segments.
        E_l_i = np.matmul(xsi.transpose(), B2_l_i)
        E_nl_i = np.matmul(xsi.transpose(), B2_nl_i)

        # STEP 3: Compute distances
        d_max = np.matmul(abs(E_l_i), z_l_i) + np.matmul(abs(E_nl_i),
                                                         z_nl_i)  # absolute distance from Uc to hyperplanes (segments of boundary)
        d_gc = abs(np.matmul(xsi.transpose(), (Uc - u0)))  # absolute projected distance from Uc to u0
        d_min[j] = d_max - d_gc  # minimum distance from u0 to hyperplanes

    if min(d_min) >= 0:
        rho = min(d_min)
    else:
        rho = -min(abs(d_min))

    return rho


def controllability_matrix(A, B):
    """
    Controllabilty matrix

    :param A: state matrix
    :param B: control matrix

    :return C: controllability matrix
    """
    n = np.shape(A)[0]
    ctrb = B
    for i in range(1, n):
        ctrb = np.hstack((ctrb, np.matmul(A ** i, B)))
    return ctrb


def objective_function_nl(delta, K, Bf_l, Bf_nl, v, lmbda=1e-4, alpha=1e-3):
    """
    Objective function for the sizing control analysis.

    :param delta: control inputs (design variables)
    :param K: sizing factors (design variables)
    :param Bf_l: linear part of the control effectiveness matrix
    :param Bf_nl: non-linear part of the control effectiveness matrix
    :param v: virtual control vector (effort) to be achieved
    :param lmbda: weight parameter on the cost (i.e., deflections minimization) objective
    :param alpha: weight parameter on the sizing objective
    :return J: objective function value
    """
    K = np.diag(K)  # transform sizing factors vector to diagonal matrix
    BfK_l = np.matmul(Bf_l, K)
    BfK_nl = np.matmul(Bf_nl, K)
    J = norm(np.matmul(BfK_l, delta) + np.matmul(BfK_nl, np.abs(delta)) - v) ** 2 + lmbda * norm(
        np.matmul(K, delta)) ** 2 + alpha * norm(K - np.identity(len(K))) ** 2
    return J


def control_optimization_nl(Bf_l, Bf_nl, v, dl_min, dl_max, dnl_min, dnl_max, delta_0, K_array=None, rho_min=0.0, lmbda=1e-4, alpha=1e-3):
    """
    Sizing control allocation

    :param Bf_l: linear part of the control effectiveness matrix
    :param Bf_nl: non linear part of the control effectiveness matrix
    :param v: virtual control vector (effort) to be achieved. Typically, v=u0
    :param dl_min: lower bounds on the control inputs
    :param dl_max: upper bounds on the control inputs
    :param dnl_min: lower bounds on the control inputs (for xacai calculation)
    :param dnl_max: upper bounds on the control inputs (for xacai calculation)
    :param delta_0: initial point for the control inputs
    :param K_array: array of 1 and 0 to define which effectors can be resized. 1 means control can be resized, 0 means resizing is forbidden (e.g., failed effector and its symmetry)
    :param rho_min: constraint on minimum XACAI value for the sized system
    :param lmbda: weight parameter on the cost (i.e., deflections minimization) objective
    :param alpha: weight parameter on the sizing objective
    :return delta_opt: optimal allocation of the controls
    :return K_opt: sizing factors
    """
    m = len(delta_0)  # number of effectors

    # Sizing factors vector definition
    K_min = 0.99 * np.ones(m)  # minimum values for sizing factors (not mandatory but helps restricting search space)
    K_max = 10 * np.ones(m)  # maximum values for sizing factors (not mandatory but helps restricting search space)
    for i, val in enumerate(K_array): # specific constraint to prohibit resizing of failed effector and its symmetry (algebraic loop)
        if val == 0:
            K_min[i] = 0.99
            K_max[i] = 1.0
    K_0 = np.ones(m)  # initial guess for sizing factors

    # Optimization variables: initial point and bounds
    # x0 = np.concatenate(((delta_min + delta_max)/2, K_0), axis=0)  # initial guess on optimization variables
    x0 = np.concatenate((delta_0, K_0), axis=0)  # initial guess on optimization variables
    x_min = np.concatenate((dl_min, K_min), axis=0)
    x_max = np.concatenate((dl_max, K_max), axis=0)
    # bounds = [x_min, x_max]  # bounds on optimization variables
    bounds = [(min_val, max_val) for min_val, max_val in zip(x_min, x_max)]  # bounds on optimization variables

    # Optimization function
    fun = lambda x: objective_function_nl(x[:m], x[m:], Bf_l, Bf_nl, v, lmbda=lmbda, alpha=alpha)

    # Constraint on XACAI value w.r.t. virtual control effort to be achieved
    rho = lambda x: xacai(np.matmul(Bf_l, np.diag(x[m:])), np.matmul(Bf_nl, np.diag(x[m:])), dl_min, dl_max, dnl_min, dnl_max, v) - rho_min
    con = {'type': 'ineq', 'fun': rho}  # for 'slsqp' optimization
    # con = NonlinearConstraint(rho, 0.0, np.inf)  # for 'trust-constr' optimization

    # Run optimization  ('slsqp' is faster but less robust than 'trust-constr')
    res = minimize(fun, method='slsqp', x0=x0, bounds=bounds, constraints=con, tol=1e-6)
    # res = minimize(fun, method='trust-constr', x0=x0, bounds=bounds, constraints=con, tol=1e-6)
    delta_opt = res.x[:m]  # optimal control inputs
    K_opt = res.x[m:]  # optimal sizing factors

    return delta_opt, K_opt


def failure_case_definition(u0, d0, dl_min, dl_max, dnl_min, dnl_max, Bf, Bf_l, Bf_nl, H):
    """
    Redefines the problem according to the failure case definition provided.

    :param u0: external perturbation vector
    :param d0: control inputs vector with the positions of the jammed effectors
    :param dl_min: lower bounds for the linear control inputs
    :param dl_max: upper bounds for the linear control inputs
    :param dnl_min: lower bounds for the non linear control inputs
    :param dnl_max: upper bounds for the non linear control inputs
    :param Bf: linearized control effectiveness matrix
    :param Bf_l: linear part of control effectiveness matrix
    :param Bf_nl: non linear part of the control effectiveness matrix
    :param H: failure matrix
    :return: the parameters of the new problem, updated according to the failure case
    """
    # step 1: update perturbations
    u0 = u0 - np.matmul(Bf_l, np.matmul(np.identity(len(H)) - H, d0)) - np.matmul(Bf_nl,
                                                                                  np.matmul(np.identity(len(H)) - H,
                                                                                            np.abs(
                                                                                                d0)))  # reference control efforts vector for failure case

    # step 2: update control bounds, e.g., freeze position of failed control(s)
    eps = 1e-6  # small relative deviation from stuck position (for numerical purpose --> see control optimization bounds)
    dl_min = np.where(np.diag(H) == 0, d0 - eps, dl_min)
    dl_max = np.where(np.diag(H) == 0, d0 + eps, dl_max)
    dnl_min = np.where(np.diag(H) == 0, d0 - eps, dnl_min)
    dnl_max = np.where(np.diag(H) == 0, d0 + eps, dnl_max)

    # step 3: update control effectiveness matrix
    Bf = np.matmul(Bf, H)  # control effectiveness matrix for failure case
    Bf_l = np.matmul(Bf_l, H)  # control effectiveness matrix for failure case (linear part)
    Bf_nl = np.matmul(Bf_nl, H)  # control effectiveness matrix for failure case (non-linear part)

    return u0, dl_min, dl_max, dnl_min, dnl_max, Bf, Bf_l, Bf_nl


def main():
    ##########################
    ### PROBLEM DEFINITION ###
    ##########################
    g = 9.81  # gravity [m/s2]

    # UAV properties
    m_uav = 1.959  # total mass [kg]
    Ix = 0.089; Iy=0.144; Iz=0.162  # moments of inertia [kg.m2]
    k_q = 0.065  # reactive torque to thrust ratio [m]
    b_w = 1.27  # wing span [m]
    c_w = 0.25  # wing mean aerodynamic chord [m]
    S_w = 0.31  # wing area [m2]
    Cd0 = 0.03  # parasitic drag coefficient [-]
    k_max_prop = 0.35  # thrust-to-weight ratio for propulsion
    k_max_vtol = 1.25  # thrust-to-weight ratio for VTOL thrust

    # Reference flight motions and perturbation efforts
    u_b0 = 19.0  # longitudinal speed [m/s]
    rho_air = 1.225  # air density [kg/m3]
    D = - 0.5 * rho_air * S_w * u_b0 ** 2 * Cd0  # drag [N]
    u0 = np.array([-D, 0, 0, 0, 0, 0])  # reference control efforts vector [X0 Y0 Z0 L0 M0 N0]'

    # Stability derivatives (e.g. Xu = 1/m * dX/du)
    Xu=-0.38; Xv=.0; Xw=0.60; Xp=.0; Xq=-0.36; Xr=.0
    Yu=.0; Yv=-0.64; Yw=.0; Yp=0.46; Yq=.0; Yr=0.79
    Zu=-0.98; Zv=.0; Zw=-10.65; Zp=0; Zq=-2.26; Zr=.0
    Lu=.0; Lv=-2.02; Lw=.0; Lp=-12.47; Lq=.0; Lr=4.05
    Mu=0.18; Mv=.0; Mw=-5.39; Mp=.0; Mq=-16.55; Mr=.0
    Nu=.0; Nv=1.30; Nw=.0; Np=0.86; Nq=.0; Nr=-3.09

    # State matrix
    A = np.array([[Xu, Xv, Xw, Xp, Xq, Xr, 0, -g, 0],
                  [Yu, Yv, Yw, Yp, Yq, Yr-u_b0, g, 0, 0],
                  [Zu, Zv, Zw, Zp, Zq+u_b0, Zr, 0, 0, 0],
                  [Lu, Lv, Lw, Lp, Lq, Lr, 0, 0, 0],
                  [Mu, Mv, Mw, Mp, Mq, Mr, 0, 0, 0],
                  [Nu, Nv, Nw, Np, Nq, Nr, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0]])

    # Control derivatives (e.g. Me = 1/Iy * dM/de)
    #Xa=-0.36/4; Ya=.0; Za=-3.62/4; La=-139.10/2; Ma=.0; Na=17.22/2  # aileron
    #Xe=-0.36/2; Ye=.0; Ze=-3.62/2; Le=-139.10/4; Me=-141.57/2; Ne=17.22/4  # elevator
    #Xr=-0.36/2; Yr=2.98; Zr=.0; Lr=6.52; Mr=.0; Nr=-26.42  # rudder
    Xa=.0; Ya=.0; Za=.0; La=-139.10/2; Ma=.0; Na=17.22/2  # aileron (longitudinal/lateral decoupling hypothesis)
    Xe=-0.36/2; Ye=.0; Ze=-3.62/2; Le=.0; Me=-141.57/2; Ne=.0  # elevator (longitudinal/lateral decoupling hypothesis)
    Xr=.0; Yr=2.98; Zr=.0; Lr=6.52; Mr=.0; Nr=-26.42  # rudder (longitudinal/lateral decoupling hypothesis)
    Xtp=k_max_prop*g; Ytp=.0; Ztp=.0; Ltp=Xtp*k_q; Mtp=.0; Ntp=.0  # propulsive propeller
    Xt=.0; Yt=.0; Zt=k_max_vtol*g/4; Lt=Zt*(b_w/4)*m_uav/Ix; Mt=Zt*(2*c_w)*m_uav/Iy; Nt=Zt*k_q  # hover propellers (thrust-to-weight ratio 1.25)

    # Control matrix
    Jf = np.diag([m_uav, m_uav, m_uav, Ix, Iy, Iz])  # matrix of inertia
    B = np.concatenate((np.linalg.inv(Jf), np.zeros((3,6))), axis=0)

    # Control effectiveness matrix
    # 1) Linearized matrix
    Bf = np.array([[Xa, Xa, Xe, Xe, Xr, Xtp, Xt, Xt, Xt, Xt],
                   [Ya, -Ya, Ye, -Ye, Yr, Ytp, Yt, -Yt, -Yt, Yt],
                   [Za, Za, Ze, Ze, Zr, Ztp, Zt, Zt, Zt, Zt],
                   [La, -La, Le, -Le, Lr, Ltp, Lt, -Lt, -Lt, Lt],
                   [Ma, Ma, Me, Me, Mr, Mtp, Mt, Mt, -Mt, -Mt],
                   [Na, -Na, Ne, -Ne, Nr, Ntp, Nt, -Nt, Nt, -Nt]
                  ])
    Bf = np.matmul(Jf, Bf)  # multiply by inertia to get our definition of Bf

    # 2) Linear efforts part
    Bf_l = np.array([[0, 0, 0, 0, 0, Xtp, Xt, Xt, Xt, Xt],
                     [Ya, -Ya, Ye, -Ye, Yr, Ytp, Yt, -Yt, -Yt, Yt],
                     [Za, Za, Ze, Ze, Zr, Ztp, Zt, Zt, Zt, Zt],
                     [La, -La, Le, -Le, Lr, Ltp, Lt, -Lt, -Lt, Lt],
                     [Ma, Ma, Me, Me, 0, Mtp, Mt, Mt, -Mt, -Mt],
                     [0, 0, 0, 0, Nr, Ntp, Nt, -Nt, Nt, -Nt]
                    ])
    Bf_l = np.matmul(Jf, Bf_l)

    # 3) Non-linear effort part
    Bf_nl = np.array([[Xa, Xa, Xe, Xe, Xr, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, Mr, 0, 0, 0, 0, 0],
                      [Na, -Na, Ne, -Ne, 0, 0, 0, 0, 0, 0]
                     ])
    Bf_nl = np.matmul(Jf, Bf_nl)

    # Effectors min-max -> delta = [ail1 ail2 elv1 elv2 rud propp prop1 prop2 prop3 prop4]'
    dl_min = np.array([-rad(25), -rad(25), -rad(25), -rad(25), -rad(25), 0, 0, 0, 0, 0])  # Linear terms
    dl_max = np.array([rad(25),  rad(25),  rad(25),  rad(25),  rad(25), 1, 1, 1, 1, 1])
    dnl_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Non linear terms
    dnl_max = np.array([rad(25),  rad(25),  rad(25),  rad(25),  rad(25), 0, 0, 0, 0, 0])

    # REDUCED PROBLEM - AXES AND EFFECTORS SELECTION
    x_idx = [0, 2, 3, 4, 5, 6, 7, 8]  # [u, w, p, q, r, theta, phi, psi]
    u_idx = [0, 3, 4, 5]  # [X, L, M, N]
    delta_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # FW-VTOL (quadplane) [ail1 ail2 elv1 elv2 rud propp prop1 prop2 prop3 prop4]
    # delta_idx = [0, 1, 2, 3, 4, 5]  # Fixed-wing [ail1 ail2 elv1 elv2 rud propp]
    u0 = np.take(u0, u_idx, axis=0)
    A = np.take(A, x_idx, axis=0); A = np.take(A, x_idx, axis=1)
    B = np.take(B, x_idx, axis=0); B = np.take(B, u_idx, axis=1)
    Bf = np.take(Bf, u_idx, axis=0); Bf = np.take(Bf, delta_idx, axis=1)
    Bf_l = np.take(Bf_l, u_idx, axis=0); Bf_l = np.take(Bf_l, delta_idx, axis=1)
    Bf_nl = np.take(Bf_nl, u_idx, axis=0); Bf_nl = np.take(Bf_nl, delta_idx, axis=1)
    dl_min = np.take(dl_min, delta_idx)
    dl_max = np.take(dl_max, delta_idx)
    dnl_min = np.take(dnl_min, delta_idx)
    dnl_max = np.take(dnl_max, delta_idx)
    n = len(A)  # problem dimension
    rho_init = xacai(Bf_l, Bf_nl, dl_min, dl_max, dnl_min, dnl_max, u0)  # xacai of initial system in nominal conditions
    print("XACAI of initial system: ", rho_init)

    ################################
    ### FAILURE CASE DEFINITION ####
    ################################
    # In this section, the user must define the failure case to evaluate.

    # Failure matrix
    H = np.diag([1, 0, 1, 1, 1, 1, 1, 1, 1, 1])  # [ail1 ail2 elv1 elv2 rud propp prop1 prop2 prop3 prop4]
    H = np.diag(np.take(np.diag(H), delta_idx))  # reduced problem

    # Array of free vs frozen sizing factors (1 means control can be resized, 0 means resizing is forbidden, e.g. for failed effector and its symmetry)
    K_array = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    # Min. XACAI for failure case, relative to initial design
    xacai_ratio = 0.5  # alpha = XACAI(failure case, resized effectors) / XACAI(nominal case, initial sizing)

    # Reference (initial) positions for failed effectors
    d0 = np.array([0, rad(25), 0, 0, 0, 0, 0, 0, 0, 0])  # jammed effectors
    # d0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # ideal failure
    d0 = np.take(d0, delta_idx)  # reduced problem

    # Update parameters according to failure case definition
    u0, dl_min, dl_max, dnl_min, dnl_max, Bf, Bf_l, Bf_nl = failure_case_definition(u0,
                                                                                    d0, dl_min, dl_max, dnl_min, dnl_max,
                                                                                    Bf, Bf_l, Bf_nl,
                                                                                    H)

    ##################################
    ### CONTROLLABILITY ASSESSMENT ###
    ##################################

    # 0) Classical controllability theory: Rank check on system with "ideal" failures
    ctrb = controllability_matrix(A, np.matmul(B, Bf))
    if matrix_rank(ctrb) == n:
        print("Rank check - Linear system (A,B*Bf) is controllable (Kalman theory).")
    else:
        print("Rank check - Linear system (A,B*Bf) is not controllable (Kalman theory).")

    # 1) XACAI on non-restricted design (unlimited oversizing of the effectors) to assess feasibility of controllability
    rho = xacai(Bf_l, Bf_nl, dl_min * 10e6, dl_max * 10e6, dnl_min * 10e6, dnl_max * 10e6, u0)
    print(r"XACAI of non-restricted system (conceptual): rho(u0,dOmega) =", rho)
    if rho > 1e-3:
        print("System might be controllable (condition 1/2). Continue process to check rank condition.")
    else:
        print("System is not controllable.")

    # 2) Sizing control allocation to derive trimmed controls and sizing factors
    v = u0  # target virtual control vector (maintain flight motion)
    rho_min = xacai_ratio * rho_init  # constraint on minimum xacai value to be achieved by sized system in failure conditions
    delta_opt, K_opt = control_optimization_nl(Bf_l, Bf_nl, v, dl_min, dl_max, dnl_min, dnl_max, d0, K_array=K_array, rho_min=rho_min, lmbda=1e-6, alpha=0.5)

    print("allocation error (rel.) =", norm(
        np.matmul(np.matmul(Bf_l, np.diag(K_opt)), delta_opt) + np.matmul(np.matmul(Bf_nl, np.diag(K_opt)),
                                                                          np.abs(delta_opt)) - v) / norm(v))
    print("control deviation (abs.) =", norm(delta_opt))
    print("optimal control inputs:", delta_opt)
    print("sizing factors:", K_opt)
    print("normalized energy norm(K*delta):", np.matmul(K_opt, abs(delta_opt)))

    # 3) Update state matrix with trimmed controls (regardless of sizing factors for simplicity)
    A_delta = np.array([[2 / u_b0 * (
                Xa * delta_opt[0] + Xa * delta_opt[1] + Xe * delta_opt[2] + Xe * delta_opt[3] + Xr * delta_opt[4]), 0,
                         0, 0, 0, 0, 0, 0, 0],
                        [2 / u_b0 * (
                                    Ya * delta_opt[0] - Ya * delta_opt[1] + Ye * delta_opt[2] - Ye * delta_opt[3] + Yr *
                                    delta_opt[4]), 0, 0, 0, 0, 0, 0, 0, 0],
                        [2 / u_b0 * (
                                    Za * delta_opt[0] + Za * delta_opt[1] + Ze * delta_opt[2] + Ze * delta_opt[3] + Zr *
                                    delta_opt[4]), 0, 0, 0, 0, 0, 0, 0, 0],
                        [2 / u_b0 * (
                                    La * delta_opt[0] - La * delta_opt[1] + Le * delta_opt[2] - Le * delta_opt[3] + Lr *
                                    delta_opt[4]), 0, 0, 0, 0, 0, 0, 0, 0],
                        [2 / u_b0 * (
                                    Ma * delta_opt[0] + Ma * delta_opt[1] + Me * delta_opt[2] + Me * delta_opt[3] + Mr *
                                    delta_opt[4]), 0, 0, 0, 0, 0, 0, 0, 0],
                        [2 / u_b0 * (
                                    Na * delta_opt[0] - Na * delta_opt[1] + Ne * delta_opt[2] - Ne * delta_opt[3] + Nr *
                                    delta_opt[4]), 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    A_delta = np.take(A_delta, x_idx, axis=0)
    A_delta = np.take(A_delta, x_idx, axis=1)  # problem reduction
    A = A + A_delta

    # 4) Rank check on controllability matrix
    # Double check XACAI value
    K = np.diag(K_opt)  # transform sizing factors vector to diagonal matrix
    rho = xacai(np.matmul(Bf_l, K), np.matmul(Bf_nl, K), dl_min, dl_max, dnl_min, dnl_max, u0)
    print(r"XACAI of resized system in failure condition: rho(u0,dOmega) =", rho)

    # Check rank of controllability matrix with updated state-space matrix
    ctrb = controllability_matrix(A, B)
    print("Problem dimension n =", n)
    print("Rank C(A,B) =", matrix_rank(ctrb))

    if rho > 0.1 and matrix_rank(ctrb) == n:
        print("System is controllable.")
    elif rho > 0.0 and matrix_rank(ctrb) == n:
        print("System in limit of controllability (XACAI close to zero)")
    else:
        print("System is not controllable.")


if __name__ == "__main__":
    main()
