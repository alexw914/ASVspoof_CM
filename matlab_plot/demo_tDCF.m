% example of use of the evaluate_tDCF_asvspoof19 function using
% the ASVspoof 2019 official baseline contermeasure systems.
% (score files stored in folder "scores")

clear; close all;

addpath(genpath('bosaris_toolkit.1.06'));
addpath('tDCF');

plot_type = Det_Plot.make_plot_window_from_string('old');
plot_obj = Det_Plot(plot_type, 'Detection error trade-off plot for ASVspoof2021LA');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 2021%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for i = 1:2
% 
%     systemi = strcat('Model', num2str(i));
%     CM_SCOREFILE = strcat('scores/', 'B0', num2str(i), '_',  track,'_primary_eval.txt');
%     
%     [~, cm_score, cm_key] = textread(CM_SCOREFILE, '%s %f %s');    
%     bona_cm     = cm_score(strcmp(cm_key, 'bonafide'));
%     spoof_cm    = cm_score(strcmp(cm_key, 'spoof'));
%     
%     [eer_cm, ~] = compute_eer(bona_cm, spoof_cm);
%     
%     
%     plot_obj.set_system(bona_cm, spoof_cm, sprintf(['LA-' systemi '(EER=%2.2f %%)'], 100* eer_cm));
%     plot_obj.plot_steppy_det({'--','LineWidth',2},' ');
%     
%     
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_obj.display_legend();