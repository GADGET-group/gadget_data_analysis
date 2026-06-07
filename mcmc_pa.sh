for i in 314  993 1723  #1.3 MeV protons
do
    python pa_mcmc.py 71 $i &
done
wait

#suspected pa events
printf "%s\n" 4007 7074 9174 11379 15302 21224 22222 28950 33414 | xargs -P 2 -I {} python pa_mcmc.py 71 {}

for i in 4434 5866 #4.33 MeV alphas
do
    python pa_mcmc.py 71 $i &
done
wait

for i in 166 563   #2 MeV alphas
do
    python pa_mcmc.py 71 $i &
done
wait