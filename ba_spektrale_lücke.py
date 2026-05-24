import numpy as np
import time
import matplotlib.pyplot as plt
from ba_implement import (power_method, rayleigh_iteration, hybrid_A, hybrid_B, hybrid_C, hybrid_D, matrix_A, starting_vector, residuum)


# Logarithmisch verteilte spektrale Lücken (von 0.0001 bis 5.0)
spectral_gaps = np.logspace(-4, np.log10(5.0), 15)  # 15 Werte von 0.0001 bis 5.0
spectral_gaps = np.round(spectral_gaps, 6)  
matrix_size = 20
max_iter = 100
eps = 0.000001
num_runs = 100  # Anzahl Durchläufe pro Matrix

# größter und kleinster Eigenwert wird fixiert
largest_eigenvalue = 12.0
smallest_eigenvalue = 1.0

# 100 Startvektoren werden erstellt (sind für alle Matrizen gleich) 
np.random.seed(42)
start_vectors = []
for i in range(num_runs):
    v = np.random.rand(matrix_size)
    v = v / np.linalg.norm(v)
    start_vectors.append(v)

results = []


for gap in spectral_gaps:
    print(f"\nTeste mit Spectral Gap: {gap}")
    
    second_eigenvalue = largest_eigenvalue - gap
    
    # Restliche Eigenwerte gleichverteilt zwischen zweitem und kleinsten
    if matrix_size > 2:
        remaining_eigenvalues = np.linspace(second_eigenvalue, smallest_eigenvalue, matrix_size - 2)
    else:
        remaining_eigenvalues = []
    
    # Eigenwerte 
    eigenvalues = np.array([largest_eigenvalue, second_eigenvalue] + list(remaining_eigenvalues))
    
    # Matrix wird erstellt 
    A, _ = matrix_A(eigenvalues)
    max_eigenvalue = eigenvalues[0]
    
    # Für jeden Algorithmus: 100 Durchläufe mit verschiedenen Startvektoren
    alg_results = {
        'Power Method': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Rayleigh Iteration': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid A': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid B': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid C': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid D': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []}
    }
    
    for v0 in start_vectors:
        # Potenzmethode
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = power_method(A, v0, max_iter, eps)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Power Method']['iterations'].append(iterations)
        alg_results['Power Method']['runtimes'].append(runtime)
        alg_results['Power Method']['residuals'].append(residual)
        alg_results['Power Method']['errors'].append(error)
      
        # RQ-Iteration
        start_time = time.time()
        eigenvalue, eigenvector, iterations = rayleigh_iteration(A, v0, max_iter, eps)
        runtime = time.time() - start_time
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Rayleigh Iteration']['iterations'].append(iterations)
        alg_results['Rayleigh Iteration']['runtimes'].append(runtime)
        alg_results['Rayleigh Iteration']['residuals'].append(residual)
        alg_results['Rayleigh Iteration']['errors'].append(error)
      
        # Hybrid A:
        start_time = time.time()
        eigenvalue, eigenvector = hybrid_A(A, v0, 3, 7, eps)
        runtime = time.time() - start_time
        iterations = 10
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Hybrid A']['iterations'].append(iterations)
        alg_results['Hybrid A']['runtimes'].append(runtime)
        alg_results['Hybrid A']['residuals'].append(residual)
        alg_results['Hybrid A']['errors'].append(error)
      
        # Hybrid B:
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = hybrid_B(A, v0, max_iter, eps, 0.001, 2)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Hybrid B']['iterations'].append(iterations)
        alg_results['Hybrid B']['runtimes'].append(runtime)
        alg_results['Hybrid B']['residuals'].append(residual)
        alg_results['Hybrid B']['errors'].append(error)
      
        # Hybrid C:
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = hybrid_C(A, v0, max_iter, eps, 3, 0.001)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Hybrid C']['iterations'].append(iterations)
        alg_results['Hybrid C']['runtimes'].append(runtime)
        alg_results['Hybrid C']['residuals'].append(residual)
        alg_results['Hybrid C']['errors'].append(error)
      
        # Hybrid D:
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = hybrid_D(A, v0, max_iter, eps, 3, 0.001)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Hybrid D']['iterations'].append(iterations)
        alg_results['Hybrid D']['runtimes'].append(runtime)
        alg_results['Hybrid D']['residuals'].append(residual)
        alg_results['Hybrid D']['errors'].append(error)
    
    # Durchschnitte und Standardabweichungen werden berechnet und in results gespeichert
    for alg_name, alg_data in alg_results.items():
        avg_iter = np.mean(alg_data['iterations'])
        avg_runtime = np.mean(alg_data['runtimes'])
        avg_residual = np.mean(alg_data['residuals'])
        avg_error = np.mean(alg_data['errors'])
        
        std_iter = np.std(alg_data['iterations'])
        std_runtime = np.std(alg_data['runtimes'])
        std_residual = np.std(alg_data['residuals'])
        std_error = np.std(alg_data['errors'])
        
        results.append({
            'algorithm': alg_name,
            'spectral_gap': gap,
            'iterations': avg_iter,
            'runtime': avg_runtime,
            'residual': avg_residual,
            'error': avg_error,
            'std_iterations': std_iter,
            'std_runtime': std_runtime,
            'std_residual': std_residual,
            'std_error': std_error
        })
        
        print(f" {alg_name}: Gap={gap:.2f}, mean {avg_iter:.1f} Iterationen, mean {avg_runtime:.4f}s, mean Residuum: {avg_residual:.2e}, mean Fehler: {avg_error:.2e}")


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

algorithms = ['Power Method', 'Rayleigh Iteration', 'Hybrid A', 'Hybrid B', 'Hybrid C', 'Hybrid D']

for alg in algorithms:
    alg_results = [r for r in results if r['algorithm'] == alg]
    if alg_results:
        gaps = [r['spectral_gap'] for r in alg_results]
        iterations = [r['iterations'] for r in alg_results]
        runtimes = [r['runtime'] for r in alg_results]
        residuals = [r['residual'] for r in alg_results]
        errors = [r['error'] for r in alg_results]
        
        std_iterations = [r['std_iterations'] for r in alg_results]
        std_runtimes = [r['std_runtime'] for r in alg_results]
        std_residuals = [r['std_residual'] for r in alg_results]
        std_errors = [r['std_error'] for r in alg_results]
        
        ax1.errorbar(gaps, iterations, yerr=std_iterations, marker='o', label=alg)
        ax2.errorbar(gaps, runtimes, yerr=std_runtimes, marker='s', label=alg)
        ax3.errorbar(gaps, residuals, yerr=std_residuals, marker='^', label=alg)
        ax4.errorbar(gaps, errors, yerr=std_errors, marker='d', label=alg)

ax1.set_xlabel('Spectral Gap')
ax1.set_ylabel('Anzahl Iterationen (Mittelwert)')
ax1.set_title('(a) Iterationen vs Spectral Gap')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, which='both')

ax2.set_xlabel('Spectral Gap')
ax2.set_ylabel('Laufzeit (s) (Mittelwert)')
ax2.set_title('(b) Laufzeit vs Spectral Gap')
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True, which='both')

ax3.set_xlabel('Spectral Gap')
ax3.set_ylabel('Residuum (Mittelwert)')
ax3.set_title('(c) Residuum vs Spectral Gap')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, which='both')

ax4.set_xlabel('Spectral Gap')
ax4.set_ylabel('Relativer Fehler (Mittelwert)')
ax4.set_title('(d) Relativer Fehler vs Spectral Gap')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend()
ax4.grid(True, which='both')

plt.tight_layout()
plt.savefig('spectral_gap_test.png')


print()
for alg in algorithms:
    alg_results = [r for r in results if r['algorithm'] == alg]
    if alg_results:
        avg_iter = np.mean([r['iterations'] for r in alg_results])
        avg_time = np.mean([r['runtime'] for r in alg_results])
        avg_residual = np.mean([r['residual'] for r in alg_results])
        avg_error = np.mean([r['error'] for r in alg_results])
        print(f"{alg}:")
        print(f" Durchschnittliche Iterationen: {avg_iter}")
        print(f" Durchschnittliche Laufzeit: {avg_time}s")
        print(f" Durchschnittliches Residuum: {avg_residual}")
        print(f" Durchschnittlicher Fehler: {avg_error}")  