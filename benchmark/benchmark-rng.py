import timeit
import numpy as np

setupNP = """
import numpy as np
"""

setupSP = """
from scipy.stats import norm, rayleigh, rice
"""

def run_test(title, test, setup):
    print '{:>40}'.format(title+':'),
    x = timeit.Timer(test, setup=setup).repeat(5,1)
    print '{:>9.3f} {:>9.3f} {:>9.3f}'.format(min(x), np.mean(x), max(x))

sizes = [
            {'title': 'Small 32x32x100 sample (flat)', 'n': 32*32*100},
            {'title': 'Small 32x32x100 sample (shaped)', 'n': '(32,32,100)'},
            {'title': 'Full 64x64x39x259 sample (flat)', 'n': 64*64*39*259},
        ]

for size in sizes:
    print '\n{:<40}'.format(size['title'])
    print '{:<41}{:>9} {:>9} {:>9}'.format('', 'min','mean','max')
    run_test('NumPy normal dist',
             "xnp = np.random.normal(scale=1.2, size={})".format(size['n']), setup=setupNP)
    run_test('NumPy rayleigh dist',
             "xnp = np.random.rayleigh(scale=1.2, size={})".format(size['n']), setup=setupNP)
    run_test('SciPy normal dist',
             "xscipy = norm.rvs(scale=1.2, size={})".format(size['n']), setup=setupSP)
    run_test('SciPy rayleigh dist',
             "xscipy = rayleigh.rvs(scale=1.2, size={})".format(size['n']), setup=setupSP)
    run_test('SciPy rice dist',
             "xscipy = rice.rvs(scale=1.2, b=1, size={})".format(size['n']), setup=setupSP)
