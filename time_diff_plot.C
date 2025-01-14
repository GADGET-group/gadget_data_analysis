void time_diff_plot();
#define _USE_MATH_DEFINES
using namespace std;

void time_diff_plot(){
    TTree *t = new TTree("t", "./runs_1-158_angles_between_all_RnPo_decays.csv");
    t->ReadFile("runs_1-158_angles_between_all_RnPo_decays.csv","angles");
    // TTree *t = new TTree("t", "./alpha_time_dist_all_angles.csv");
    // t->ReadFile("alpha_time_dist_all_angles.csv","time_diff");
    // TTree *t = new TTree("t", "./alpha_time_dist_160_deg_and_up.csv");
    // t->ReadFile("alpha_time_dist_160_deg_and_up.csv","time_diff");
    t->Draw("angles");
}