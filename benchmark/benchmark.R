#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(neuRosim))
library(rbenchmark)
library(optparse)

print_test <- function(title, results, repeats=10) {
    cat(sprintf('%60s:',title))
    cat(sprintf('%9.3f %9d',results[1,'elapsed']/repeats,repeats),'\n')

}

print_header <- function(title) {
    cat('\n',title,'\n')
    cat(sprintf('%60s %9s %9s','','mean','reps'),'\n')

}

really_slow <- 1
slow <- 3
normal <- 10

benchmark_stimfunction <- function() {
    print_header('Stimfunction benchmarks')

    b <- benchmark(replications=normal,
                   stimfunction(400, c(2, 12), 2, .1))
    print_test('Few events, 200 TRs',b,normal)

    b <- benchmark(replications=normal,
                 expression(s <- stimfunction(400, c(1,2,3,4,5,6,7,8,9,10,11,12,13,14), 2, .1)))
    print_test('More events, 200 TRs',b,normal)

    onsets <- seq(1995)
    b <- benchmark(replications=normal,
                   stimfunction(4000, onsets, 2, .1))
    print_test('More events, 2000 TRs',b,normal)
}

benchmark_specifydesign <- function() {
    print_header('Specifydesign benchmarks')

    for (cond in c('none','gamma','double-gamma')) {
        total_time = 400
        onsets = list(c(0, 30, 60), c(25,75))
        dur = list(10, 3)
        effectsize = list(3, 10)
        acc = .5
        b <- benchmark(replications=normal,
                s <- specifydesign(onsets, dur, total_time, 2, effectsize, acc, conv=cond))
        print_test(paste('2 conds, 200 TRs, few events,',cond),b,normal)

        total_time = 4000
        onsets = list(c(0, 30, 60), c(550,650))
        b <- benchmark(replications=normal,
                s <- specifydesign(onsets, dur, total_time, 2, effectsize, acc, conv=cond))
        print_test(paste('2 conds, 2000 TRs, few events,',cond),b,normal)

        onsets = list(seq(1,979,20), seq(2,979,19))
        b <- benchmark(replications=normal,
                s <- specifydesign(onsets, dur, total_time, 2, effectsize, acc, conv=cond))
        print_test(paste('2 conds, 2000 TRs, many events,',cond),b,normal)

        onsets = list(seq(1,979,20), seq(2,979,19),
                      seq(3,979,18), seq(4,979,17),
                      seq(5,979,16), seq(6,979,15),
                      seq(7,979,14), seq(8,979,13),
                      seq(9,979,12), seq(10,979,11))
        dur = list(10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
        effectsize = list(3, 10, 2, 6, 7, 1, 5, 4, 9, 8)
        b <- benchmark(replications=normal,
                s <- specifydesign(onsets, dur, total_time, 2, effectsize, acc, conv=cond))
        print_test(paste('10 conds, 2000 TRs, many events,',cond) ,b,normal)
    }
}

benchmark_specifyregion <- function() {
    print_header('Specifyregion benchmarks')

    for (shape in c('cube','sphere','manual')) {
        f <- function() {
            out <- array(0,dim=c(10,10))
            out <- out + specifyregion(dim=c(10,10), coord=rbind(c(1,1)), radius=1, form=shape, fading=0)
            out <- out + specifyregion(dim=c(10,10), coord=rbind(c(7,7)), radius=2, form=shape, fading=0)
        }
        b <- benchmark(replications=normal,
                       f())
        print_test(paste('Toy space [10x10], 2 regions,',shape),b,normal)

        f <- function() {
            out <- array(0,dim=c(256,256))
            out <- out + specifyregion(dim=c(256,256), coord=rbind(c(1,1)), radius=13, form=shape, fading=0)
            out <- out + specifyregion(dim=c(256,256), coord=rbind(c(240,240)), radius=22, form=shape, fading=0)
        }
        b <- benchmark(replications=normal,
                       f())
        print_test(paste('Highres Slice [256x256], 2 regions,',shape),b,normal)

        for (fade in c(0, .5, 1)) {
            if (fade > 0 & shape == 'manual') {
                cat(sprintf('%60s:',paste('Highres Slice [256x256], 50 regions, fade ',fade,', ',shape,sep='')))
                cat(sprintf('%9s','skipped'),'\n')
            } else {
                coords <- array(sample.int(256,100, replace=T),dim=c(50,2))
                radii <- sample.int(20,50, replace=T)
                f <- function() {
                    out <- array(0,dim=c(256,256))
                    for (i in 1:50) {
                        out <- out + specifyregion(dim=c(256,256), coord=rbind(coords[i,]), radius=radii[i], form=shape, fading=fade)
                    }
                }
                b <- benchmark(replications=slow,
                               f())
                print_test(paste('Highres Slice [256x256], 50 regions, fade ',fade,', ',shape,sep=''),b,slow)
            }
        }

        for (fade in c(0, .5, 1)) {
            if (fade > 0 & shape == 'manual') {
                cat(sprintf('%60s:',paste('Wholebrain [64x64x32], 50 regions, fade ',fade,', ',shape,sep='')))
                cat(sprintf('%9s','skipped'),'\n')
            } else {
                coords <- array(sample.int(32,150,replace=T),dim=c(50,3))
                radii <- sample.int(20,50, replace=T)
                f <- function() {
                    out <- array(0,dim=c(64,64,32))
                    for (i in 1:50) {
                        out <- out + specifyregion(dim=c(64,64,32), coord=rbind(coords[i,]), radius=radii[i], form=shape, fading=fade)
                    }
                }
                b <- benchmark(replications=really_slow,
                               f())
                print_test(paste('Wholebrain [64x64x32], 50 regions, fade ',fade,', ',shape,sep=''),b,really_slow)
            }
        }
    }
}

benchmark_systemnoise <- function() {
    print_header('System noise benchmarks')

    for (type in c('gaussian','rician')) {
        b <- benchmark(replications=normal,
                       systemnoise(dim=c(1),nscan=200, sigma=1.5, type=type))
        print_test(paste('Single voxel, 200 TRs,',type),b,normal)

        b <- benchmark(replications=normal,
                       systemnoise(dim=c(1),nscan=200000, sigma=1.5, type=type))
        print_test(paste('Single voxel, 200000 TRs,',type),b,normal)

        b <- benchmark(replications=normal,
                       systemnoise(dim=c(64,64,32),nscan=200, sigma=1.5, type=type))
        print_test(paste('Wholebrain [64x64x32], 200 TRs,',type),b,really_slow)
    }
}

benchmark_lowfreqnoise <- function() {
    print_header('Low frequency noise benchmarks')

    b <- benchmark(replications=normal,
                   lowfreqdrift(dim=c(1),nscan=200, freq=128.0, TR=2))
    print_test('Single voxel, 200 TRs',b,normal)

    b <- benchmark(replications=normal,
                   lowfreqdrift(dim=c(1),nscan=2000, freq=128.0, TR=2))
    print_test('Single voxel, 2000 TRs',b,normal)

    b <- benchmark(replications=normal,
                   lowfreqdrift(dim=c(64,64),nscan=200, freq=128.0, TR=2))
    print_test('Whole slice [64x64], 200 TRs',b,normal)

    b <- benchmark(replications=normal,
                   lowfreqdrift(dim=c(64,64,32),nscan=200, freq=128.0, TR=2))
    print_test('Wholebrain [64x64x32], 200 TRs',b,normal)
}

benchmark_physnoise <- function() {
    print_header('Physiological noise benchmarks')

    b <- benchmark(replications=normal,
                   physnoise(dim=c(1),nscan=200, sigma=1.5, TR=2))
    print_test('Single voxel, 200 TRs',b,normal)

    b <- benchmark(replications=normal,
                   physnoise(dim=c(1),nscan=2000, sigma=1.5, TR=2))
    print_test('Single voxel, 2000 TRs',b,normal)

    b <- benchmark(replications=normal,
                   physnoise(dim=c(64,64,32),nscan=200, sigma=1.5, TR=2))
    print_test('Wholebrain [64x64x32], 200 TRs',b,normal)
}

benchmark_tasknoise <- function() {
    print_header('Task-related noise benchmarks')

    total_time = 400
    onsets = list(c(0, 30, 60), c(25,75))
    dur = list(10, 3)
    effectsize = list(3, 10)
    acc = .5
    s <- specifydesign(onsets, dur, total_time, 2, effectsize, acc, conv='double-gamma')
    b <- benchmark(replications=normal,
                   tasknoise(s, sigma=1.5))
    print_test('Single voxel, 200 TRs',b,normal)

    total_time = 4000
    onsets = list(c(0, 30, 60), c(550,650))
    s <- specifydesign(onsets, dur, total_time, 2, effectsize, acc, conv='double-gamma')
    b <- benchmark(replications=normal,
                   tasknoise(s, sigma=1.5))
    print_test('Single voxel, 2000 TRs',b,normal)

    total_time = 400
    onsets = list(c(0, 30, 60), c(25,75))
    s <- specifydesign(onsets, dur, total_time, 2, effectsize, acc, conv='double-gamma')
    s <- array(s,dim=c(64,64,32,200))

    b <- benchmark(replications=slow,
                   tasknoise(s, sigma=1.5))
    print_test('Wholebrain [64x64x32], 200 TRs',b,slow)

}

benchmark_temporalnoise <- function() {
    print_header('Temporally correlated noise benchmarks')

    b <- benchmark(replications=normal,
                   temporalnoise(dim=c(1),nscan=200, rho=.2, sigma=1.5))
    print_test('Single voxel, 200 TRs, AR(1)',b,normal)

    b <- benchmark(replications=normal,
                   temporalnoise(dim=c(1),nscan=200, rho=c(.3, .2, .4), sigma=1.5))
    print_test('Single voxel, 200 TRs, AR(3)',b,normal)

    b <- benchmark(replications=normal,
                   temporalnoise(dim=c(1),nscan=2000, rho=.2, sigma=1.5))
    print_test('Single voxel, 2000 TRs, AR(1)',b,normal)

    b <- benchmark(replications=normal,
                   temporalnoise(dim=c(1),nscan=2000, rho=c(.3, .2, .4), sigma=1.5))
    print_test('Single voxel, 2000 TRs, AR(3)',b,normal)

    b <- benchmark(replications=slow,
                   temporalnoise(dim=c(64,64),nscan=200, rho=.2, sigma=1.5))
    print_test('Whole slice [64x64], 200 TRs, AR(1)',b,slow)

    b <- benchmark(replications=slow,
                   temporalnoise(dim=c(64,64),nscan=200, rho=c(.3, .2, .4), sigma=1.5))
    print_test('Whole slice [64x64], 200 TRs, AR(3)',b,slow)

    b <- benchmark(replications=really_slow,
                   temporalnoise(dim=c(64,64,32),nscan=200, rho=.2, sigma=1.5))
    print_test('Wholebrain [64x64x32], 200 TRs, AR(1)',b,really_slow)

    b <- benchmark(replications=really_slow,
                   temporalnoise(dim=c(64,64,32),nscan=200, rho=c(.3, .2, .4), sigma=1.5))
    print_test('Wholebrain [64x64x32], 200 TRs, AR(3)',b,really_slow)

}

benchmark_spatialnoise<- function() {
    print_header('Spatial noise benchmarks')

    b <- benchmark(replications=normal,
                   spatialnoise(dim=c(10,10),nscan=1, rho=.7, sigma=1.5))
    print_test('Toy dim [10x10], 1 TR',b,normal)

    b <- benchmark(replications=normal,
                   spatialnoise(dim=c(64,64),nscan=1, rho=.7, sigma=1.5))
    print_test('Whole slice [64x64], 1 TR',b,normal)

    b <- benchmark(replications=normal,
                   spatialnoise(dim=c(256,256),nscan=1, rho=.7, sigma=1.5))
    print_test('Highres slice [256x256], 1 TR',b,normal)

    b <- benchmark(replications=normal,
                   spatialnoise(dim=c(10,10,10),nscan=1, rho=.7, sigma=1.5))
    print_test('Toy dim [10x10x10], 1 TR',b,normal)

    b <- benchmark(replications=slow,
                   spatialnoise(dim=c(256,256,128),nscan=1, rho=.7, sigma=1.5))
    print_test('Highres [256x256x128], 1 TR',b,slow)

    b <- benchmark(replications=normal,
                   spatialnoise(dim=c(10,10),nscan=200, rho=.7, sigma=1.5))
    print_test('Toy dim [10x10], 200 TR',b,normal)

    b <- benchmark(replications=slow,
                   spatialnoise(dim=c(64,64),nscan=200, rho=.7, sigma=1.5))
    print_test('Whole slice [64x64], 200 TR',b,slow)

    b <- benchmark(replications=really_slow,
                   spatialnoise(dim=c(64,64,32),nscan=200, rho=.7, sigma=1.5))
    print_test('Wholebrain [64x64x32], 200 TR',b,really_slow)
}

benchmark_simTS<- function() {
    print_header('Sim TS benchmarks')

    f <- function(type) {
        d <- simprepTemporal(totaltime=400, onsets=list(c(1,41,81,121,161),c(15,55,95,135,175)),
                             durations=list(20,7), TR=2, effectsize=list(1,1),hrf='double-gamma')
        simTSfmri(design=d, base=0, nscan=200, TR=2, noise=type, SNR=1)
    }

    for (noise in c('none','white','temporal','low-frequency','physiological','task-related')) 
    {
        b <- benchmark(replications=normal,
                       f(noise))
        print_test(paste('Single Voxel, 200 TRs,',noise),b,normal)
    }

    f <- function() {
        d <- simprepTemporal(totaltime=400, onsets=list(c(1,41,81,121,161),c(15,55,95,135,175)),
                             durations=list(20,7), TR=2, effectsize=list(1,1),hrf='double-gamma')
        simTSfmri(design=d, base=0, nscan=200, TR=2, noise='mixture',w=c(.2,.2,.2,.2,.2), SNR=1)
    }
    b <- benchmark(replications=normal,
                   f())
    print_test('Single Voxel, 200 TRs, mixture',b,normal)

    f <- function(type) {
        d <- simprepTemporal(totaltime=4000, onsets=list(c(1,41,81,121,161),c(15,55,95,135,175)),
                             durations=list(20,7), TR=2, effectsize=list(1,1),hrf='double-gamma')
        simTSfmri(design=d, base=0, nscan=2000, TR=2, noise=type, SNR=1)
    }

    for (noise in c('none','white','temporal','low-frequency','physiological','task-related')) 
    {
        b <- benchmark(replications=normal,
                       f(noise))
        print_test(paste('Single Voxel, 2000 TRs,',noise),b,normal)
    }

    f <- function() {
        d <- simprepTemporal(totaltime=4000, onsets=list(c(1,41,81,121,161),c(15,55,95,135,175)),
                             durations=list(20,7), TR=2, effectsize=list(1,1),hrf='double-gamma')
        simTSfmri(design=d, base=0, nscan=2000, TR=2, noise='mixture',w=c(.2,.2,.2,.2,.2), SNR=1)
    }
    b <- benchmark(replications=normal,
                   f())
    print_test('Single Voxel, 2000 TRs, mixture',b,normal)
}

benchmark_simVOL<- function() {
    print_header('Sim VOL benchmarks')

    f <- function(type) {
        d <- simprepTemporal(totaltime=400, onsets=list(c(1,41,81,121,161),c(15,55,95,135,175)),
                             durations=list(20,7), TR=2, effectsize=list(1,1),hrf='double-gamma')
        i <- simprepSpatial(regions=3,list(c(1,1), c(5,5),c(6,0)),radius=c(1,2,1),form='cube', fading=c(.5, 0, 0))
        simVOLfmri(design=d, image=i, dim=c(10,10), base=0, nscan=200, TR=2, noise=type, SNR=2)
    }

    for (noise in c('none','white','temporal','low-frequency','physiological','task-related','spatial'))
    {
        b <- benchmark(replications=normal,
                       f(noise))
        print_test(paste('Single Voxel, 200 TRs,',noise),b,normal)
    }

    f <- function(type) {
        d <- simprepTemporal(totaltime=400, onsets=list(c(1,41,81,121,161),c(15,55,95,135,175)),
                             durations=list(20,7), TR=2, effectsize=list(1,1),hrf='double-gamma')
        i <- simprepSpatial(regions=3,list(c(1,1), c(5,5),c(6,0)),radius=c(1,2,1),form='cube', fading=c(.5, 0, 0))
        simVOLfmri(design=d, image=i, dim=c(10,10), base=0, nscan=200, TR=2, noise=type, w=c(.2, .2, .2, .1, .1, .2), SNR=2)
    }

    noise <- 'mixture'
    b <- benchmark(replications=normal,
                   f(noise))
    print_test(paste('Single Voxel, 200 TRs,',noise),b,normal)

    f <- function(type) {
        d <- simprepTemporal(totaltime=400, onsets=list(c(1,41,81,121,161),c(15,55,95,135,175)),
                             durations=list(20,7), TR=2, effectsize=list(1,1),hrf='double-gamma')
        i <- simprepSpatial(regions=3,list(c(1,1), c(5,5),c(6,0)),radius=c(1,2,1),form='cube', fading=c(.5, 0, 0))
        simVOLfmri(design=d, image=i, dim=c(64,64), base=0, nscan=200, TR=2, noise=type, SNR=2)
    }

    for (noise in c('none','white','temporal','low-frequency','physiological','task-related','spatial'))
    {
        b <- benchmark(replications=slow,
                       f(noise))
        print_test(paste('Whole slice [64x64], 200 TRs,',noise),b,slow)
    }

    f <- function(type) {
        d <- simprepTemporal(totaltime=400, onsets=list(c(1,41,81,121,161),c(15,55,95,135,175)),
                             durations=list(20,7), TR=2, effectsize=list(1,1),hrf='double-gamma')
        i <- simprepSpatial(regions=3,list(c(1,1), c(5,5),c(6,0)),radius=c(1,2,1),form='cube', fading=c(.5, 0, 0))
        simVOLfmri(design=d, image=i, dim=c(64,64), base=0, nscan=200, TR=2, noise=type, w=c(.2, .2, .2, .1, .1, .2), SNR=2)
    }

    noise <- 'mixture'
    b <- benchmark(replications=slow,
                   f(noise))
    print_test(paste('Whole slice [64x64], 200 TRs,',noise),b,slow)

    f <- function(type) {
        d <- simprepTemporal(totaltime=400, onsets=list(c(1,41,81,121,161),c(15,55,95,135,175)),
                             durations=list(20,7), TR=2, effectsize=list(1,1),hrf='double-gamma')
        i <- simprepSpatial(regions=3,list(c(1,1,1), c(5,5,5),c(6,0,0)),radius=c(1,2,1),form='cube', fading=c(.5, 0, 0))
        simVOLfmri(design=d, image=i, dim=c(64,64,32), base=0, nscan=200, TR=2, noise=type, SNR=2)
    }

    for (noise in c('none','white','temporal','low-frequency','physiological','task-related','spatial'))
    {
        b <- benchmark(replications=really_slow,
                       f(noise))
        print_test(paste('Whole slice [64x64x32], 200 TRs,',noise),b,really_slow)
    }

    f <- function(type) {
        d <- simprepTemporal(totaltime=400, onsets=list(c(1,41,81,121,161),c(15,55,95,135,175)),
                             durations=list(20,7), TR=2, effectsize=list(1,1),hrf='double-gamma')
        i <- simprepSpatial(regions=3,list(c(1,1,1), c(5,5,5),c(6,0,0)),radius=c(1,2,1),form='cube', fading=c(.5, 0, 0))
        simVOLfmri(design=d, image=i, dim=c(64,64,32), base=0, nscan=200, TR=2, noise=type, w=c(.2, .2, .2, .1, .1, .2), SNR=2)
    }

    noise <- 'mixture'
    b <- benchmark(replications=really_slow,
                   f(noise))
    print_test(paste('Whole slice [64x64x32], 200 TRs,',noise),b,really_slow)
}

option_list <- list(
    make_option(c('-a','--all'), action="store_true", default=F, help="Run the full suite of benchmarks"),
    make_option(c('-v','--sim-vol'), action="store_true", default=F, help="Run the simVOL benchmarks"),
    make_option(c('-t','--sim-ts'), action="store_true", default=F, help="Run the simTS benchmarks"),
    make_option(c('-n','--noise'), action="store_true", default=F, help="Run the noise benchmarks"),
    make_option('--stimfunc', action="store_true", default=F, help="Run the stimfunction benchmarks"),
    make_option('--specdesign', action="store_true", default=F, help="Run the specifydesign benchmarks"),
    make_option('--specregion', action="store_true", default=F, help="Run the specifyregion benchmarks"),
    make_option('--systemnoise', action="store_true", default=F, help="Run the systemnoise benchmarks"),
    make_option('--lowfreq', action="store_true", default=F, help="Run the lowfreqnoise benchmarks"),
    make_option('--phys', action="store_true", default=F, help="Run the physnoise benchmarks"),
    make_option('--task-related', action="store_true", default=F, help="Run the task-related noise benchmarks"),
    make_option('--temporal', action="store_true", default=F, help="Run the temporally correlated noise benchmarks"),
    make_option('--spatial', action="store_true", default=F, help="Run the spatially correlated noise benchmarks")
)

opt <- parse_args(OptionParser(option_list=option_list, description="Run BOLDsim benchmarks.",
                  epilogue="Be default no benchmarks will be run, so make sure to specify one!"))

if (opt$all) {
    for (name in names(opt)) {
        opt[[name]] = T
    }
}
if (opt$noise) {
    for (name in c('systemnoise','lowfreq','phys','task-related','temporal','spatial')) {
        opt[[name]] = T
    }
}

if (opt$stimfunc)
    benchmark_stimfunction()

if (opt$specdesign)
    benchmark_specifydesign()

if (opt$specregion)
    benchmark_specifyregion()

if (opt$systemnoise)
    benchmark_systemnoise()

if (opt$lowfreq)
    benchmark_lowfreqnoise()

if (opt$phys)
    benchmark_physnoise()

if (opt$`task-related`)
    benchmark_tasknoise()

if (opt$temporal)
    benchmark_temporalnoise()

if (opt$spatial)
    benchmark_spatialnoise()

if (opt$`sim-ts`)
    benchmark_simTS()

if (opt$`sim-vol`)
    benchmark_simVOL()
