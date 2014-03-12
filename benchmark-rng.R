library(neuRosim)
library(VGAM)

run_test <- function(title, test) {
    cat(sprintf('%30s:',title))
    x <- replicate(5,system.time(eval(test)))
    cat(sprintf('%9.3f %9.3f %9.3f',min(x[3,]),mean(x[3,]),max(x[3,])),'\n')

}

sizes = list(
             list(title='Small 32x32x100 sample', n=32*32*100),
             list(title='Full 64*64*39*259 sample', n=64*64*39*259)
             )

for (i in 1:length(sizes)) {
    cat('\n', sizes[[i]][['title']], '\n')
    cat(sprintf('%30s ',''))
    cat(sprintf('%9s %9s %9s','min','mean','max'),'\n')
    run_test('Builtin normal dist',
             expression(y <- rnorm(n=sizes[[i]][['n']], sd=1.2)))
    run_test('VGAM rayleigh dist',
             expression(y <- rrayleigh(n=sizes[[i]][['n']], scale=1.2)))
    run_test('VGAM rice dist',
             expression(y <- VGAM::rrice(n=sizes[[i]][['n']], vee=1, sigma=1.2)))
    run_test('neuRosim rice dist',
             expression(y <- neuRosim::rrice(n=sizes[[i]][['n']], vee=1, sigma=1.2)))
}
