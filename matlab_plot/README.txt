A MATLAB code package for computing t-DCF and EER metrics for ASVspoof2019.
(Version 2.0)

The main difference with regard to Version 1.0 is the use of a revisited formulation
of the tandem detection cost function. Further details can be found in the paper:

T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
Audio, Speech and Language Processing (TASLP).

USAGE:
Run evaluate_tDCF_asvspoof19.m to compute the metrics as follows:

	evaluate_tDCF_asvspoof19(CM_SCOREFILE, ASV_SCOREFILE, LEGACY)

where 

  CM_SCOREFILE points to your own countermeasure score file.
  ASV_SCOREFILE points to the organizers' ASV scores.
  LEGACY is a boolean flag. If set to true, the t-DCF formulation
        employed in the ASVspoof 2019 challenge is used (discouraged).
  
  NOTE! There are two different ASV score files provided by the organizers:
        One for the physical access (PA) scenario and the other for the logical access (LA) scenario.
        Be sure to point to the right ASV score file according to the scenario (LA or PA).

A demo script "demo_tDCF.m" is provided. It computes the normalized minimum t-DCF
   of the baseline systems for the LA and PA scenarios.
