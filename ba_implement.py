import numpy as np
from scipy.linalg import solve, lu_factor, lu_solve
import time

def power_method(A, v0, maxIter, eps):
    v = v0 / np.linalg.norm(v0)
    rayleighs = []

    for _ in range(maxIter):

        v_k = A @ v
        v_new = v_k / np.linalg.norm(v_k)
        e_value = v_new.T @ A @ v_new
        rayleighs.append(e_value)

        if np.linalg.norm(A @ v_new - e_value * v_new) < eps:
            v = v_new
            break
        
        v = v_new

    return e_value, v, rayleighs



def rayleigh_iteration(A, v0, maxIter, eps):
    v = v0 / np.linalg.norm(v0)
    e0 = v.T @ A @ v
    iterations = 0

    for i in range(maxIter):
        iterations = i + 1
        v_k = solve(A - e0 * np.identity(A.shape[0]), v)
        v_new = v_k / np.linalg.norm(v_k)

        e_k = v_new.T @ A @ v_new
        if np.linalg.norm(A @ v_new - e_k * v_new) < eps:
            e0 = e_k
            v = v_new
            break
        e0 = e_k
        v = v_new
    
    return e0, v, iterations



# Hybrid A:
# Feste Iterationen Power-Method und dann RQ-Iteration
def hybrid_A(A, v0, maxPower, maxRayleigh, eps):
    v = v0 / np.linalg.norm(v0)

    for _ in range(maxPower):
        v_k = A @ v
        v = v_k / np.linalg.norm(v_k)

    e_value = v.T @ A @ v

    for _ in range(maxRayleigh):
        v_k = solve(A - e_value * np.identity(A.shape[0]), v)
        v_new = v_k / np.linalg.norm(v_k)
        e_k = v_new.T @ A @ v_new
        if np.linalg.norm(A @ v_new - e_k * v_new) < eps:
            break
        e_value = e_k
        v = v_new
    
    return e_value, v


# Hybrid B:
# beginnt mit Power-Method und speichert in jeder Iteration den Rayleigh-Quotient in einer Liste. 
# Damit wird ein Differenzmaß berechnet und wenn dieses unter ein Threshold fällt, wird auf RQ-Iteration umgeschaltet.
# In beiden Phasen Abbruchkriterum mit Norm.
def hybrid_B(A, v0, maxIter, eps, threshold, rayleigh_steps):
    v = v0 / np.linalg.norm(v0)

    rayleighs = []

    for i in range(maxIter):
        v_k = A @ v
        v_new = v_k / np.linalg.norm(v_k)
        e_value = v_new.T @ A @ v_new
        rayleighs.append(e_value)

        if len(rayleighs) >= rayleigh_steps:
            gap = sorted(rayleighs[-rayleigh_steps:], reverse=True)
            spectral_gap = gap[0] - gap[1]
        
            if spectral_gap < threshold:
                e0 = e_value

                for _ in range(maxIter - i):
                    v_k = solve(A - e0 * np.identity(A.shape[0]), v_new)
                    v_new2 = v_k / np.linalg.norm(v_k)
                    e_k = v_new2.T @ A @ v_new2

                    if np.linalg.norm(A @ v_new2 - e_k * v_new2) < eps:
                        return e_k, v_new2, rayleighs
                
                    e0 = e_k
                    v_new = v_new2
                return e0, v_new, rayleighs
        
        if np.linalg.norm(A @ v_new - e_value * v_new) < eps:
            return e_value, v_new, rayleighs
        v = v_new

    return e_value, v, rayleighs


# Hybrid C:
# Beginn mit Power Method.
# Hier wird nur für alle n Steps, der RQ berechnet, wobei geprüft wird, ob RQ zum letzten unter Threshold fällt bzw. auch mit Norm. 
# Wird Konvergenz erreicht, wird auf RQ-Iteration geschalten.
def hybrid_C(A, v0, maxIter, eps, n_steps, threshold):
    v = v0 / np.linalg.norm(v0)
    rayleighs = []
    last_rayleigh = None

    for i in range(maxIter):
        v_k = A @ v
        v_new = v_k / np.linalg.norm(v_k)

        if i % n_steps == 0 or i == maxIter - 1:
            e_value = v_new.T @ A @ v_new
            rayleighs.append(e_value)
            if last_rayleigh is not None:
                if abs(e_value - last_rayleigh) < threshold or np.linalg.norm(A @ v_new - e_value * v_new) < eps:
                    e0 = e_value
                    for j in range(maxIter- i):
                        v_k = solve(A - e0 * np.identity(A.shape[0]), v_new)
                        v_new2 = v_k / np.linalg.norm(v_k)
                        e_k = v_new2.T @ A @ v_new2
                        if np.linalg.norm(A @ v_new2 - e_k * v_new2) < eps:
                            return e_k, v_new2, rayleighs
                        
                        e0 = e_k
                        v_new = v_new2
                    return e0, v_new, rayleighs
            last_rayleigh = e_value

        v = v_new

    return e_value, v, rayleighs


# Hybrid D:
# Wie Hybrid C, aber mit festem Shift in der Rayleigh-Phase.
# Verwendet LR-Zerlegung für effizientere Lösung des LGS.
def hybrid_D(A, v0, maxIter, eps, n_steps, threshold):
    v = v0 / np.linalg.norm(v0)
    rayleighs = []
    last_rayleigh = None

    for i in range(maxIter):
        v_k = A @ v
        v_new = v_k / np.linalg.norm(v_k)

        if i % n_steps == 0 or i == maxIter - 1:
            e_value = v_new.T @ A @ v_new
            rayleighs.append(e_value)
            if last_rayleigh is not None:
                if abs(e_value - last_rayleigh) < threshold or np.linalg.norm(A @ v_new - e_value * v_new) < eps:
                    # konstanter Shift
                    mu = e_value
                    
                    # Einmalige LR-Zerlegung 
                    LR = lu_factor(A - mu * np.identity(A.shape[0]))
                    
                    for j in range(maxIter - i):
                        # LR-Zerlegung
                        v_k = lu_solve(LR, v_new)
                        v_new2 = v_k / np.linalg.norm(v_k)
                        e_k = v_new2.T @ A @ v_new2
                        
                        if np.linalg.norm(A @ v_new2 - e_k * v_new2) < eps:
                            return e_k, v_new2, rayleighs
                        
                        v_new = v_new2
                    return e_k, v_new, rayleighs
            last_rayleigh = e_value

        v = v_new

    return e_value, v, rayleighs
    





def matrix_A(eigenvalues):
    n = len(eigenvalues)
    q, r = np.linalg.qr(np.random.randn(n,n))
    A = q @ np.diag(eigenvalues) @ q.T
    return A, q

def starting_vector(eigenvalues):
    vector = np.random.rand(len(eigenvalues))
    #vector = np.ones(len(eigenvalues))
    #vector = np.zeros(len(eigenvalues))
    #vector[0] = 1
    return vector

def residuum(A, v, e_value):
    return np.linalg.norm(A @ v - e_value * v)



def main():
    maxIter = 100
    maxPower = 3
    maxRayleigh = 100
    threshold = 0.001
    rayleigh_steps = 2
    n_steps = 3
    eps = 0.0000000001

    ev = [12, 11.9, 7, 6, 5.5, 5, 4, 3, 2, 1]

    A = matrix_A(ev)
    v = starting_vector(ev)
    #v = q[:, 0]
    #A = np.array([[2,1,7,4], [1,2,3,3], [3,4,1,9], [4,4,6,0]])
    #v = np.array([4,1,2,0])

    start = time.time()
    power_value, power_vector, power_rayleigh = power_method(A[0], v, maxIter, eps)
    end = time.time()
    print(f"Runtime Power-Method: {end-start:.6f} sec")

    start = time.time()
    rayleigh_value, rayleigh_vector, rayleigh_iterations = rayleigh_iteration(A[0], v, maxIter, eps)
    end = time.time()
    print(f"Runtime RQ-Iteration: {end-start:.6f} sec")

    start = time.time()
    hybrid_A_value, hybrid_A_vector = hybrid_A(A[0], v, maxPower, maxRayleigh, eps)
    end = time.time()
    print(f"Runtime Hybrid-A: {end-start:.6f} sec")

    start = time.time()
    hybrid_B_value, hybrid_B_vector, hybrid_B_rayleighs = hybrid_B(A[0], v, maxIter, eps, threshold, rayleigh_steps)
    end = time.time()
    print(f"Runtime Hybrid-B: {end-start:.6f} sec")

    start = time.time()
    hybrid_C_value, hybrid_C_vector, hybrid_C_rayleighs = hybrid_C(A[0], v, maxIter, eps, n_steps, threshold)
    end = time.time()
    print(f"Runtime Hybrid-C: {end-start:.6f} sec")

    start = time.time()
    hybrid_D_value, hybrid_D_vector, hybrid_D_rayleighs = hybrid_D(A[0], v, maxIter, eps, n_steps, threshold)
    end = time.time()
    print(f"Runtime Hybrid-D: {end-start:.6f} sec")


    eigenvalues = np.linalg.eigvals(A[0])
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    spectral_gap = eigenvalues_sorted[0] - eigenvalues_sorted[1]


    print(f"Eigenvalues: {eigenvalues_sorted}")
    print(f"Spectral Gap: {spectral_gap}")
    print()
    for i, rayl in enumerate(power_rayleigh):
        print(f"Interation {i+1}: {rayl}")
    print()
    print(f"Eigenvectors: {matrix_A(ev)[1]}")
    print()
    print(f"Power-Method: Eigenvalue: {power_value}, Eigenvector: {power_vector}, rayleighs: {power_rayleigh}, residuum: {residuum(A[0], power_vector, power_value)}")
    print()
    print(f"RQ-Iteration: Eigenvalue: {rayleigh_value}, Eigenvector: {rayleigh_vector}, residuum: {residuum(A[0], rayleigh_vector, rayleigh_value)}")
    print()
    print(f"Hybrid-A: Eigenvalue: {hybrid_A_value}, Eigenvector: {hybrid_A_vector}, residuum: {residuum(A[0], hybrid_A_vector, hybrid_A_value)}")
    print()
    print(f"Hybrid-B: Eigenvalue: {hybrid_B_value}, Eigenvector: {hybrid_B_vector}, rayleighs: {hybrid_B_rayleighs}, residuum: {residuum(A[0], hybrid_B_vector, hybrid_B_value)}")
    print()
    print(f"Hybrid-C: Eigenvalue: {hybrid_C_value}, Eigenvector: {hybrid_C_vector}, rayleighs: {hybrid_C_rayleighs}, residuum: {residuum(A[0], hybrid_C_vector, hybrid_C_value)}")
    print()
    print(f"Hybrid-D: Eigenvalue: {hybrid_D_value}, Eigenvector: {hybrid_D_vector}, rayleighs: {hybrid_D_rayleighs}, residuum: {residuum(A[0], hybrid_D_vector, hybrid_D_value)}")
    

if __name__ == "__main__":
    main()

