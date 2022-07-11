function evaluate_tDCF_asvspoof19(CM_SCOREFILE, ASV_SCOREFILE, legacy)

% Set t-DCF parameters
if legacy
    cost_model.Pspoof       = 0.05;  % Prior probability of a spoofing attack
    cost_model.Ptar         = (1 - cost_model.Pspoof) * 0.99; % Prior probability of target speaker
    cost_model.Pnon         = (1 - cost_model.Pspoof) * 0.01; % Prior probability of nontarget speaker
    cost_model.Cmiss_asv    = 1;     % Cost of ASV system falsely rejecting target speaker
    cost_model.Cfa_asv      = 10;    % Cost of ASV system falsely accepting nontarget speaker
    cost_model.Cmiss_cm     = 1;     % Cost of CM system falsely rejecting target speaker
    cost_model.Cfa_cm       = 10;    % Cost of CM system falsely accepting spoof
else
    cost_model.Pspoof       = 0.05;  % Prior probability of a spoofing attack
    cost_model.Ptar         = (1 - cost_model.Pspoof) * 0.99; % Prior probability of target speaker
    cost_model.Pnon         = (1 - cost_model.Pspoof) * 0.01; % Prior probability of nontarget speaker
    cost_model.Cmiss        = 1;     % Cost of tandem system falsely rejecting target speaker
    cost_model.Cfa          = 10;    % Cost of tandem system falsely accepting nontarget speaker
    cost_model.Cfa_spoof    = 10;    % Cost of tandem system falsely accepting spoof
end

% Load organizer's ASV scores
[~, ~, ~, ~, asv_key, asv_score] = textread(ASV_SCOREFILE, '%s %s %s %s %s %f');

% Load CM scores (replace these with the scores of your detectors).
[~, cm_key, cm_score] = textread(CM_SCOREFILE, '%s %s %f');

% Extract target, nontarget and spoof scores from the ASV scores
tar_asv     = asv_score(strcmp(asv_key, 'target'))';
non_asv     = asv_score(strcmp(asv_key, 'nontarget'))';
spoof_asv   = asv_score(strcmp(asv_key, 'spoof'))';

% Extract bona fide (real human) and spoof scores from the CM scores
bona_cm     = cm_score(strcmp(cm_key, 'bonafide'));
spoof_cm    = cm_score(strcmp(cm_key, 'spoof'));

% Fix ASV operating point to EER threshold
[eer_asv, asv_threshold] = compute_eer(tar_asv, non_asv);

% Visualize ASV and CM scores
% figure;
% subplot(121);
% [h, xi] = ksdensity(tar_asv);   plot(xi, h, 'gr'); hold on;
% [h, xi] = ksdensity(non_asv);   plot(xi, h, 'k'); 
% [h, xi] = ksdensity(spoof_asv); plot(xi, h, 're');
% h = plot(asv_threshold, 0, 'mo'); set(h, 'MarkerSize', 10);
% legend('Target', 'Nontarget', 'Spoof', 'EER threshold');
% xlabel('ASV score'); ylabel('PDF'); 
% title('ASV scores', 'FontSize', 12, 'FontWeight', 'bold');
% 
% subplot(122);
% [h, xi] = ksdensity(bona_cm);   plot(xi, h, 'gr'); hold on;
% [h, xi] = ksdensity(spoof_cm);  plot(xi, h, 're'); 
% legend('Bona fide', 'Spoof');
% xlabel('CM score'); ylabel('PDF');
% title('CM scores', 'FontSize', 12, 'FontWeight', 'bold');

% Obtain the detection error rates of the ASV system
[Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold);

% Equal error rate of the countermeasure
[eer_cm, ~] = compute_eer(bona_cm, spoof_cm);

% Compute t-DCF
if legacy
    [tDCF_curve, CM_thresholds] = compute_tDCF_legacy(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, true);
else
    [tDCF_curve, CM_thresholds] = compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, true);
end

% Minimum normalized t-DCF and the corresponding threshold
[min_tDCF, argmin_tDCF] = min(tDCF_curve); 
min_tDCF_threshold = CM_thresholds(argmin_tDCF);

% compute DET of CM and get Pmiss and Pfa for the selected threshold t_CM
[Pmiss_cm, Pfa_cm, CM_thresholds] = compute_det_curve(bona_cm, spoof_cm);
Pmiss_t_CM = Pmiss_cm(CM_thresholds == min_tDCF_threshold);
Pfa_t_CM = Pfa_cm(CM_thresholds == min_tDCF_threshold);

% fprintf('ASV SYSTEM\n');
% fprintf('     EER               \t= %5.5f %%\t\t Equal error rate (target vs. nontarget discrimination)\n',   100 * eer_asv);
% fprintf('     Pfa               \t= %5.5f %%\t\t False acceptance rate of nontargets\n',   100 * Pfa_asv);
% fprintf('     Pmiss             \t= %5.5f %%\t\t Miss (false rejection) rate of targets\n',   100 * Pmiss_asv);
% if legacy
%     fprintf('     1-Pmiss,spoof     \t= %5.5f %%\t Spoof false acceptance rate ("NOT miss spoof trial")\n\n', 100 * (1 - Pmiss_spoof_asv));
% else
%     fprintf('     Pfa,spoof         \t= %5.5f %%\t\t Spoof false acceptance rate\n\n', 100 * Pfa_spoof_asv);
% end
% 
% fprintf('CM SYSTEM\n');
% fprintf('     EER                  \t= %5.5f %%\t\t Equal error rate from CM scores pooled across all attacks \n', 100 * eer_cm);
% fprintf('     Pfa(t_CM_min_tDCF)   \t= %5.5f %%\t\t False acceptance rate of spoofs \n', 100 * Pfa_t_CM);
% fprintf('     Pmiss(t_CM_min_tDCF) \t= %5.5f %%\t\t Miss (false rejection) rate of bonafide \n\n', 100 * Pmiss_t_CM);
% 
% 
% fprintf('TANDEM\n');
% fprintf('     min-tDCF          \t= %5.5f\n',   min_tDCF);

% Plot t-DCF as function of the CM threshold. 


% figure;
% plot(CM_thresholds, tDCF_curve); hold on;
% h = plot(min_tDCF_threshold, min_tDCF, 'mo'); set(h, 'MarkerSize', 10);
% xlabel('CM threshold (operating point)');
% ylabel('Norm t-DCF');
% title('Normalized t-DCF', 'FontSize', 12, 'FontWeight', 'bold');
% h = line([min(CM_thresholds) max(CM_thresholds)], [1 1]); set(h, 'LineStyle', '--', 'Color', 'k');
% legend('t-DCF', sprintf('min t-DCF (%5.5f)', min_tDCF), 'Arbitrarily bad CM (Norm t-DCF=1)');
% axis([min(CM_thresholds) max(CM_thresholds) 0 1.5]);

% Traditional DET plot using Bosaris toolkit
plot_type = Det_Plot.make_plot_window_from_string('old');
plot_obj = Det_Plot(plot_type, 'Detection error trade-off plot of CM');
plot_obj.set_system(bona_cm, spoof_cm, sprintf('CM system (EER=%2.2f %%)', 100* eer_cm));
plot_obj.plot_steppy_det({'b','LineWidth',1},' ');
plot_obj.display_legend();


end