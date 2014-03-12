#!/usr/bin/env python

import numpy as np
import boldsim.sim as sim
import timeit
import argparse

parser = argparse.ArgumentParser(description='Run BOLDsim benchmarks.',
                                 epilog='By default no benchmarks will be run, so make sure to specify one!')
parser.add_argument('-a', '--all', action='store_true', help='Run the full suite of benchmarks')
parser.add_argument('-v','--sim-vol', action='store_true', help='Run the simVOL benchmarks')
parser.add_argument('-t','--sim-ts', action='store_true', help='Run the simTS benchmarks')
parser.add_argument('-n', '--noise', action='store_true', help='Run the full suite of noise benchmarks')
parser.add_argument('--stimfunc', action='store_true', help='Run the stimfunction benchmarks')
parser.add_argument('--specdesign', action='store_true', help='Run the specifydesign benchmarks')
parser.add_argument('--specregion', action='store_true', help='Run the specifyregion benchmarks')
parser.add_argument('--systemnoise', action='store_true', help='Run the systemnoise benchmarks')
parser.add_argument('--lowfreq', action='store_true', help='Run the lowfreqnoise benchmarks')
parser.add_argument('--phys', action='store_true', help='Run the physnoise benchmarks')
parser.add_argument('--task-related', action='store_true', help='Run the task-related noise benchmarks')
parser.add_argument('--temporal', action='store_true', help='Run the temporally correlated noise benchmarks')
parser.add_argument('--spatial', action='store_true', help='Run the spatially correlated noise benchmarks')

def run_test(title, test, setup, repeats=10):
    print '{:>60}'.format(title+':'),
    x = timeit.Timer(test, setup=setup).repeat(repeats,1)
    print '{:>9.3f} {:>9.3f} {:>9.3f} {:9d}'.format(min(x), np.mean(x), max(x), repeats)

def print_header(title):
    print '\n{:<60}'.format(title)
    print '{:<61}{:>9} {:>9} {:>9} {:>9}'.format('', 'min','mean','max','reps')

setup = """
import numpy as np
from boldsim import sim
"""

really_slow = 1
slow = 3
normal = 10

#################

def benchmark_stimfunction():
    print_header('Stimfunction benchmarks')

    run_test('Few events, 200 TRs',
            "s = sim.stimfunction(400, [2, 12], 2, .1)",
            setup=setup)
    run_test('More events, 200 TRs',
            "s = sim.stimfunction(400, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], 2, .1)",
            setup=setup)
    run_test('More events, 2000 TRs',
            "s = sim.stimfunction(4000, onsets, 2, .1)",
            setup=setup + """
onsets = np.arange(1995)
    """)

##################

def benchmark_specifydesign():

    print_header('Specifydesign benchmarks')

    setup_short_few = setup + """
total_time = 400
onsets = [[0, 30, 60], [25, 75]]
dur = [[10],[3]]
effect_sizes = [3, 10]
acc = .5
    """

    setup_long_few = setup + """
total_time = 4000
onsets = [[0, 30, 60], [550, 650]]
dur = [[10],[3]]
effect_sizes = [3, 10]
acc = .5
    """

    setup_long_many = setup + """
total_time = 4000
onsets = [np.arange(1,979,20), np.arange(2,979,19)]
dur = [[10],[3]]
effect_sizes = [3, 10]
acc = .5
    """

    setup_long_many_conds = setup + """
total_time = 4000
onsets = [np.arange(1,979,20), np.arange(2,979,19),
          np.arange(3,979,18), np.arange(4,979,17),
          np.arange(5,979,16), np.arange(6,979,15),
          np.arange(7,979,14), np.arange(8,979,13),
          np.arange(9,979,12), np.arange(10,979,11),
         ]
dur = [[10],[9],[8],[7],[6],[5],[4],[3],[2],[1]]
effect_sizes = [3, 10, 2, 6, 7, 1, 5, 4, 9, 8]
acc = .5
    """

    titles = ['2 conds, 200 TRs, few events, {}',
              '2 conds, 2000 TRs, few events, {}',
              '2 conds, 2000 TRs, many events, {}',
              '10 conds, 2000 TRs, many events, {}',
              ]
    setups = [setup_short_few, setup_long_few, setup_long_many, setup_long_many_conds]

    for hrf in ['none','gamma','double-gamma']:
        for title, specific_setup in zip(titles, setups):
            run_test(title.format(hrf),
                    """d = sim.specifydesign(total_time, onsets=onsets, durations=dur, accuracy=acc,
                                          effect_sizes=effect_sizes, TR=2, conv='{}')""".format(hrf),
                    setup=specific_setup, repeats=normal)

def benchmark_specifyregion():
    print_header('Specifyregion benchmarks')

    for shape in ['cube','sphere','manual']:
        run_test('Toy space [10x10], 2 regions, {}'.format(shape),
                """s = sim.specifyregion(dim=[10,10],coord=[[1,1],[7,7]],radius=[1,2],
                                         form='{}', fading=0)""".format(shape),
                setup=setup, repeats=normal)

        run_test('Highres Slice [256x256], 2 regions, {}'.format(shape),
                """s = sim.specifyregion(dim=[256,256],coord=[[1,1],[240,240]],radius=[13,22],
                                         form='{}', fading=0)""".format(shape),
                setup=setup, repeats=normal)

        for fade in [0, .5, 1]:
            run_test('Highres slice [256x256], 50 regions, fade {}, {}'.format(fade, shape),
                    """s = sim.specifyregion(dim=[256,256],coord=coords,radius=radii,
                                             form='{}', fading={})""".format(shape,fade),
                    repeats=slow, setup=setup + """
coords = [[np.random.randint(low=0, high=255),
               np.random.randint(low=0, high=255)] for x in range(50)]
radii = [np.random.randint(low=1, high=20) for x in range(50)]
                    """)

        for fade in [0, .5, 1]:
            run_test('Wholebrain [64x64x32], 50 regions, fade {}, {}'.format(fade, shape),
                    """s = sim.specifyregion(dim=[64,64,32],coord=coords,radius=radii,
                                             form='{}', fading={})""".format(shape, fade),
                    repeats=really_slow, setup=setup + """
coords = [[np.random.randint(low=0, high=63),
           np.random.randint(low=0, high=63),
           np.random.randint(low=0, high=31)] for x in range(50)]
radii = [np.random.randint(low=1, high=20) for x in range(50)]""")


def benchmark_systemnoise():
    print_header('System noise benchmarks')

    for noise_type in ['gaussian','rayleigh']:
        run_test('Single voxel, 200 TRs, {}'.format(noise_type),
                """s = sim.system_noise(nscan=200, sigma=1.5,
                                        dim=(1,), noise_dist='{}')""".format(noise_type),
                setup=setup, repeats=normal)
        run_test('Single voxel, 200000 TRs, {}'.format(noise_type),
                """s = sim.system_noise(nscan=200000, sigma=1.5,
                                        dim=(1,), noise_dist='{}')""".format(noise_type),
                setup=setup, repeats=normal)

        run_test('Wholebrain voxels [64x64x32], 200 TRs, {}'.format(noise_type),
                """s = sim.system_noise(nscan=200, sigma=1.5,
                                        dim=(64,64,32), noise_dist='{}')""".format(noise_type),
                setup=setup, repeats=slow)

def benchmark_lowfreqnoise():
    print_header('Low frequency noise benchmarks')

    run_test('Single voxel, 200 TRs',
            'noise = sim.lowfreqdrift(nscan=200, freq=128.0, dim=(1,))',
            setup=setup, repeats=normal)
    run_test('Single voxel, 2000 TRs',
            'noise = sim.lowfreqdrift(nscan=2000, freq=128.0, dim=(1,))',
            setup=setup, repeats=normal)
    run_test('Wholebrain [64x64x32], 200 TRs',
            'noise = sim.lowfreqdrift(nscan=200, freq=128.0, dim=(64,64,32))',
            setup=setup, repeats=slow)

def benchmark_physnoise():
    print_header('Physiological noise benchmarks')

    run_test('Single voxel, 200 TRs',
            'noise = sim.physnoise(nscan=200, dim=(1,))',
            setup=setup)
    run_test('Single voxel, 2000 TRs',
            'noise = sim.physnoise(nscan=2000, dim=(1,))',
            setup=setup)
    run_test('Wholebrain [64x64x32], 200 TRs',
            'noise = sim.physnoise(nscan=200, dim=(64,64,32))',
            setup=setup)

def benchmark_tasknoise():
    print_header('Task-related noise benchmarks')

    run_test('Single voxel, 200 TRs',
            'noise = sim.tasknoise(design=d, dim=(1,))',
            repeats=normal, setup=setup+"""
total_time = 400
onsets = [[0, 30, 60], [25, 75]]
dur = [[1],[3]]
effect_sizes = [1, 2]
acc = .5

d = sim.specifydesign(total_time, onsets=onsets, durations=dur, accuracy=acc,
                          effect_sizes=effect_sizes, conv='double-gamma', TR=2)
            """)
    run_test('Single voxel, 2000 TRs',
            'noise = sim.tasknoise(design=d, dim=(1,))',
            repeats=normal, setup=setup+"""
total_time = 4000
onsets = [[0, 30, 60], [550, 650]]
dur = [[10],[3]]
effect_sizes = [3, 10]
acc = .5

d = sim.specifydesign(total_time, onsets=onsets, durations=dur, accuracy=acc,
                          effect_sizes=effect_sizes, conv='double-gamma', TR=2)
            """)
    run_test('Wholebrain [64x64x32], 200 TRs',
            'noise = sim.tasknoise(design=d, dim=(64,64,32))',
            repeats=slow, setup=setup+"""
total_time = 400
onsets = [np.arange(1,370,20), np.arange(2,379,19),
          np.arange(3,370,18), np.arange(4,379,17),
          np.arange(5,379,16), np.arange(6,379,15),
          np.arange(7,379,14), np.arange(8,379,13),
          np.arange(9,379,12), np.arange(10,379,11),
         ]
dur = [[10],[9],[8],[7],[6],[5],[4],[3],[2],[1]]
effect_sizes = [3, 10, 2, 6, 7, 1, 5, 4, 9, 8]
acc = .5

d = sim.specifydesign(total_time, onsets=onsets, durations=dur, accuracy=acc,
                          effect_sizes=effect_sizes, conv='double-gamma', TR=2)
            """)

def benchmark_temporalnoise():
    print_header('Temporally correlated noise benchmarks')

    run_test('Single Voxel, 200 TRs, AR(1)',
            'noise = sim.temporalnoise(dim=(1,),nscan=200, ar_coef=[.2])',
            setup=setup, repeats=normal)
    run_test('Single Voxel, 200 TRs, AR(3)',
            'noise = sim.temporalnoise(dim=(1,),nscan=200, ar_coef=[.3, .2, .4])',
            setup=setup, repeats=normal)
    run_test('Single Voxel, 2000 TRs, AR(1)',
            'noise = sim.temporalnoise(dim=(1,),nscan=2000, ar_coef=[.2])',
            setup=setup, repeats=normal)
    run_test('Single Voxel, 2000 TRs, AR(3)',
            'noise = sim.temporalnoise(dim=(1,),nscan=2000, ar_coef=[.3, .2, .4])',
            setup=setup, repeats=normal)
    run_test('Whole slice [64x64], 200 TRs, AR(1)',
            'noise = sim.temporalnoise(dim=(64,64),nscan=200, ar_coef=[.2])',
            setup=setup, repeats=slow)
    run_test('Whole slice [64x64x32], 200 TRs, AR(3)',
            'noise = sim.temporalnoise(dim=(64,64),nscan=200, ar_coef=[.3, .2, .4])',
            setup=setup, repeats=slow)
    run_test('Wholebrain [64x64x32], 200 TRs, AR(1)',
            'noise = sim.temporalnoise(dim=(64,64,32),nscan=200, ar_coef=[.2])',
            setup=setup, repeats=really_slow)
    run_test('Wholebrain [64x64x32], 200 TRs, AR(3)',
            'noise = sim.temporalnoise(dim=(64,64,32),nscan=200, ar_coef=[.3, .2, .4])',
            setup=setup, repeats=really_slow)

def benchmark_spatialnoise():
    print_header('Spatial noise benchmarks')

    run_test('Toy dim [10x10], 1 TR',
            'noise = sim.spatialnoise(dim=(10,10),nscan=1,rho=.7)',
            setup=setup, repeats=normal)
    run_test('Whole slice [64x64], 1 TR',
            'noise = sim.spatialnoise(dim=(64,64),nscan=1,rho=.7)',
            setup=setup, repeats=normal)
    run_test('Hires slice [256x256], 1 TR',
            'noise = sim.spatialnoise(dim=(256,256),nscan=1,rho=.7)',
            setup=setup, repeats=normal)
    run_test('Toy dim [10x10x10], 1 TR',
            'noise = sim.spatialnoise(dim=(10,10,10),nscan=1,rho=.7)',
            setup=setup, repeats=normal)
    run_test('Hires [256x256x128], 1 TR',
            'noise = sim.spatialnoise(dim=(256,256,128),nscan=1,rho=.7)',
            setup=setup, repeats=slow)
    run_test('Toy dim [10x10], 200 TRs',
            'noise = sim.spatialnoise(dim=(10,10),nscan=200,rho=.7)',
            setup=setup, repeats=normal)
    run_test('Whole slice [64x64], 200 TRs',
            'noise = sim.spatialnoise(dim=(64,64),nscan=200,rho=.7)',
            setup=setup, repeats=slow)
    run_test('Wholebrain [64x64x32], 200 TRs',
            'noise = sim.spatialnoise(dim=(64,64,32),nscan=200,rho=.7)',
            setup=setup, repeats=really_slow)

def benchmark_simTS():
    print_header('Sim TS benchmarks')

    for noise_type in ['none','white','temporal','low-freq','phys','task-related','mixture']:
        run_test('Single Voxel, 200 TRs, {}'.format(noise_type),
                """
design = sim.simprepTemporal(total_time=400,onsets=[[1,41, 81, 121, 161],
                                                    [15, 55, 95, 135, 175]],
                             durations=[[20],[7]],
                             effect_sizes=[1], conv='double-gamma')

ts = sim.simTSfmri(design, noise='{}')""".format(noise_type),
                setup=setup, repeats=normal)

    for noise_type in ['none','white','temporal','low-freq','phys','task-related','mixture']:
        run_test('Single Voxel, 2000 TRs, {}'.format(noise_type),
                """
design = sim.simprepTemporal(total_time=4000,onsets=[[1,41, 81, 121, 161],
                                                    [15, 55, 95, 135, 175]],
                             durations=[[20],[7]],
                             effect_sizes=[1], conv='double-gamma')

ts = sim.simTSfmri(design, noise='{}')""".format(noise_type),
                setup=setup, repeats=normal)

def benchmark_simVOL():
    print_header('Sim VOL benchmarks')

    for noise_type in ['none','white','temporal','low-freq','phys','task-related','mixture']:
        run_test('Toy Dim [10x10], 200 TRs, {}'.format(noise_type),
                """
design = sim.simprepTemporal(total_time=400,onsets=[[1,41, 81, 121, 161],
                                                    [15, 55, 95, 135, 175]],
                                         durations=[[20],[7]], TR=2,
                                         effect_sizes=[1], conv='double-gamma')

image = sim.simprepSpatial(regions=3,
                          coord=[[1,1],[5,5],[6,0]],
                          radius=[1,2,0],
                           form=['cube','sphere','cube'],
                           fading=[.5, 0, 0])

sim_ds = sim.simVOLfmri(designs=design,
                           images=image,
                           noise='{}',
                            base=10,
                            dim=[10,10],
                            SNR=2)
""".format(noise_type),
                setup=setup, repeats=normal)

    for noise_type in ['none','white','temporal','low-freq','phys','task-related','mixture']:
        run_test('Whole slice [64x64], 200 TRs, {}'.format(noise_type),
                """
design = sim.simprepTemporal(total_time=400,onsets=[[1,41, 81, 121, 161],
                                                    [15, 55, 95, 135, 175]],
                                         durations=[[20],[7]], TR=2,
                                         effect_sizes=[1], conv='double-gamma')

image = sim.simprepSpatial(regions=3,
                          coord=[[1,1],[5,5],[6,0]],
                          radius=[1,2,0],
                           form=['cube','sphere','cube'],
                           fading=[.5, 0, 0])

sim_ds = sim.simVOLfmri(designs=design,
                           images=image,
                           noise='{}',
                            base=10,
                            dim=[64,64],
                            SNR=2)
""".format(noise_type),
                setup=setup, repeats=slow)
    for noise_type in ['none','white','temporal','low-freq','phys','task-related','spatial','mixture']:
        run_test('Wholebrain [64x64x32], 200 TRs, {}'.format(noise_type),
                """
design = sim.simprepTemporal(total_time=400,onsets=[[1,41, 81, 121, 161],
                                                    [15, 55, 95, 135, 175]],
                                         durations=[[20],[7]], TR=2,
                                         effect_sizes=[1], conv='double-gamma')

image = sim.simprepSpatial(regions=3,
                          coord=[[1,1,1],[5,5,5],[6,0,0]],
                          radius=[1,2,0],
                           form=['cube','sphere','cube'],
                           fading=[.5, 0, 0])

sim_ds = sim.simVOLfmri(designs=design,
                           images=image,
                           noise='{}',
                            base=10,
                            dim=[64,64,32],
                            SNR=2)
""".format(noise_type),
                setup=setup, repeats=really_slow)

def run_benchmarks(args):
    if args['stimfunc']:
        benchmark_stimfunction()
    if args['specdesign']:
        benchmark_specifydesign()
    if args['specregion']:
        benchmark_specifyregion()
    if args['systemnoise']:
        benchmark_systemnoise()
    if args['lowfreq']:
        benchmark_lowfreqnoise()
    if args['phys']:
        benchmark_physnoise()
    if args['task_related']:
        benchmark_tasknoise()
    if args['temporal']:
        benchmark_temporalnoise()
    if args['spatial']:
        benchmark_spatialnoise()
    if args['sim_ts']:
        benchmark_simTS()
    if args['sim_vol']:
        benchmark_simVOL()

if __name__ == "__main__":
    args = parser.parse_args()
    args = vars(args)
    if args['all']:
        for key in args.keys():
            args[key] = True
    if args['noise']:
        for key in ['systemnoise','lowfreq','phys','task_related',
                'temporal','spatial']:
            args[key] = True
    run_benchmarks(args)

