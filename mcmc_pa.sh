#suspected pa events
printf "%s\n" 4007 7074 9174 11379 15302 21224 22222 28950 33414 | xargs -P 3 -I {} python pa_mcmc.py 71 {}

#1.3 MeV protons
printf "%s\n" 314 993 1723 | xargs -P 3 -I {} python pa_mcmc.py 71 {}

#4.3 MeV alpha + recoil
printf "%s\n" 592 4165 4583 | xargs -P 3 -I {} python pa_mcmc.py 71 {}


#4.33 MeV alphas (no recoil, won't fit well)
# printf "%s\n" 4434 5866  | xargs -P 3 -I {} python pa_mcmc.py 71 {}

# #2 MeV alphas (no recoil)
# printf "%s\n" 166 563  | xargs -P 3 -I {} python pa_mcmc.py 71 {}

