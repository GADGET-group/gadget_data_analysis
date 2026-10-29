#dim, num_events, threshold, counts_per_event, sigma_min, sigma_max, gain_mu, gain_sigma, min_dist_to_edge, device
dim=20
sigma_min=1.5
sigma_max=3
min_dist_to_edge=3
gain_mu=1.0
gain_sigma=0.2

screen_name="1k_t0_sig${gain_sigma}"
echo $screen_name
screen -S $screen_name -d -m
screen -S $screen_name -X stuff "conda activate e21072 \n"
screen -S $screen_name -X stuff "python toy_model.py $dim 1000 0 1000 $sigma_min $sigma_max $gain_mu $gain_sigma $min_dist_to_edge 0 \n"

screen_name="10k_t0_sig${gain_sigma}"
echo $screen_name
screen -S $screen_name -d -m
screen -S $screen_name -X stuff "conda activate e21072 \n"
screen -S $screen_name -X stuff "python toy_model.py $dim 10000 0 1000 $sigma_min $sigma_max $gain_mu $gain_sigma $min_dist_to_edge 0 \n"

screen_name="100k_t0_sig${gain_sigma}"
echo $screen_name
screen -S $screen_name -d -m
screen -S $screen_name -X stuff "conda activate e21072 \n"
screen -S $screen_name -X stuff "python toy_model.py $dim 100000 0 1000 $sigma_min $sigma_max $gain_mu $gain_sigma $min_dist_to_edge 2 \n"

screen_name="1k_t0.5_sig${gain_sigma}"
echo $screen_name
screen -S $screen_name -d -m
screen -S $screen_name -X stuff "conda activate e21072 \n"
screen -S $screen_name -X stuff "python toy_model.py $dim 1000 0.5 1000 $sigma_min $sigma_max $gain_mu $gain_sigma $min_dist_to_edge 1 \n"

screen_name="10k_t0.5_sig${gain_sigma}"
echo $screen_name
screen -S $screen_name -d -m
screen -S $screen_name -X stuff "conda activate e21072 \n"
screen -S $screen_name -X stuff "python toy_model.py $dim 10000 0.5 1000 $sigma_min $sigma_max $gain_mu $gain_sigma $min_dist_to_edge 1 \n"

screen_name="100k_t0.5_sig${gain_sigma}"
echo $screen_name
screen -S $screen_name -d -m
screen -S $screen_name -X stuff "conda activate e21072 \n"
screen -S $screen_name -X stuff "python toy_model.py $dim 100000 0.5 1000 $sigma_min $sigma_max $gain_mu $gain_sigma $min_dist_to_edge 3 \n"