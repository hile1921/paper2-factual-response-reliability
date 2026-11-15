function run_paper_figures(ids)
% Six main figures. Uses saved observations and frozen estimates only.
if nargin==0,ids=1:6;end
root=fileparts(mfilename('fullpath'));addpath(fullfile(root,'src'));addpath(fullfile(root,'design'));
set(groot,'defaultTextInterpreter','none','defaultAxesTickLabelInterpreter','none','defaultLegendInterpreter','none');
L=load(fullfile(root,'data','figure_inputs.mat'));D=L.D;
A=load(fullfile(root,'data','design_inputs.mat'));
Q=load(fullfile(root,'data','context_distribution.mat'));D.context_dist=Q.context_dist;
D.case_c1=A.case_c1;D.case_c2=A.case_c2;D.case_c3=A.case_c3;D.c2.rho_ci=A.rho_ci;
if any(ids==5),validate_context_plot_data(D);end
out=fullfile(root,'output');if ~exist(out,'dir'),mkdir(out);end
jobs={@paper01_overview,@paper02_training,@paper03_atlas,@paper04_alignment,@paper05_context,@paper06_contrasts};
names={'P01_study_design','P02_training_responses','P03_matched_facts','P04_preference_answer_link','P05_context_cases_and_outcomes','P06_context_effects'};
reports=struct('figure',{},'status',{},'MATLAB',{});
for k=ids
 fprintf('Rendering %s\n',names{k});
 try
  f=jobs{k}(D);edfig.export(f,out,names{k});
  if k==5
   f=paper05_condition_distributions(D);edfig.export(f,out,'P05b_country_effects');
  end
  reports(end+1)=struct('figure',names{k},'status','complete','MATLAB',version);
 catch err
  reports(end+1)=struct('figure',names{k},'status',err.message,'MATLAB',version);
  fid=fopen(fullfile(out,'DESIGN_MANIFEST.json'),'w','n','UTF-8');fprintf(fid,'%s',jsonencode(reports,PrettyPrint=true));fclose(fid);rethrow(err);
 end
end
fid=fopen(fullfile(out,'DESIGN_MANIFEST.json'),'w','n','UTF-8');fprintf(fid,'%s',jsonencode(reports,PrettyPrint=true));fclose(fid);
fprintf('Completed %d evidence-design figures: %s\n',numel(ids),out);
end
