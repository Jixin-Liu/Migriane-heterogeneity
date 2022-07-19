function [mySignifLVs,Lx,V]=myPLSC(X,Y,CONST_BEHAV_NAMES,diagnosis_grouping,CONST_DIAGNOSIS,outputPath)
%% Partial Least Squares (PLS) for Neuroimaging %%
% Main script

% Behavior PLS : looks for optimal associations between imaging and
% behavior data. Imaging can be either a volume (voxel-based)
% (e.g., brain activity) or a functional correlation matrix. 
% If input is a volume, a binary mask should be entered so that all
% subjects have the same number of voxels.

% Requires SPM for loading & reading volumes

% ~ PLS steps ~
% 1. Data normalization
% 2. Cross-covariance matrix
% 3. Singular value decomposition
% 4. Brain & behavior scores
% 5. Permutation testing for LV significance
% 6. Bootstrapping to test stability of brain saliences
% 7. Contribution of original variables to the LVs

% ~ FIGURES ~
% I.   Screeplot (explained covariance)
% II.  Correlations between brain & behavior scores
% III. Brain saliences (bootstrap ratio)
% IV. Behavior saliences
% V.  Brain structure coefficients
% VI. Behavior structure coefficients


% NOTE FOR DATASETS WITH SUBJECTS FROM DIFFERENT GROUPS 
% (e.g., controls & patients) : it is possible to normalize data 
% within each group instead of across subjects 
% (options 1,3 in myPLS_norm & change subj_grouping to diagnosis_grouping)
% Note that permutations and bootstrapping should be done within each 
% group, rather than across all subjects.

%% Set parameters
display = false; 
% Paths
Time = 1;       % ÖØ¸´²âÁ¿´ÎÊý

% Indicate type of imaging data
imagingType = 'volume' ; % 'volume' (voxel-based) or 'corrMat' (correlation matrix)

% Data normalization options (% default=1 - zscore across all subjects)
CONST_NORM = 1;     % ¹éÒ»»¯µÄ·½Ê½
%CONST_NORM  = 2;%·Ö×é

% Permutations & Bootstrapping
NUM_PERMS = 1000;           % ÖÃ»»¼ìÑé´ÎÊý
NUM_BOOTSTRAP = 1000;       % bootstrap´ÎÊý

resultsFilename = 'myPLSresults'; % name of results file that will be saved in outputPath

%% Load data
NUM_GROUPS = length(unique(diagnosis_grouping));    % ÊäÈëÊý¾ÝµÄ×éÊý
%maskFile = spm_vol(fullfile('ROI1FCMap_1211.nii')); % filename of binary mask that will constrain analysis 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check if X and Y matrices have same number of rows (subjects)
if size(X,1) ~= size(Y,1)
    disp('Matrices X and Y should have same number of rows [number of samples]');
end

% Get number of subjects
CONST_NUM_SUBJ = size(X,1); % number of subjects
CONST_NUM_IMAGING = size(X,2);  % number of imaging measures
CONST_NUM_BEHAV = size(Y,2);% number of behavior measures
subj_grouping = ones(CONST_NUM_SUBJ,1);
%% 1. Normalize X and Y

% Save original matrices
X0 = X; Y0 = Y;

X = myPLS_norm(X,NUM_GROUPS,diagnosis_grouping,CONST_NORM);
Y = myPLS_norm(Y,NUM_GROUPS,diagnosis_grouping,CONST_NORM);

%% 2. Cross-covariance matrix
clear R

R = myPLS_cov(X,Y,CONST_NORM,diagnosis_grouping);

%% 3. Singular value decomposition
clear U S V

[U,S,V] = svd(R,'econ');
NUM_LVs = min(size(S));

% ICA convention: turn latent variables (LVs) such that max is positive
for iter_lv = 1 : NUM_LVs
    [~,idx] = max(abs(V(:,iter_lv)));
    if sign(V(idx,iter_lv)) < 0
        V(:,iter_lv) = -V(:,iter_lv);
        U(:,iter_lv) = -U(:,iter_lv);
    end;
end;

explVarLVs = (diag(S).^2) / sum(diag(S.^2)); % Explained covariance by each LV

%% 4. Brain and behavior scores
clear Lx Ly

switch CONST_NORM
    case 1
        Lx = X * V; % Brain scores : original imaging data projected on brain saliences
        Ly = Y * U; % Behavior scores : original behavior data projected on behavior saliences
    case 2
        Lx = X * V;
        for i = 1: NUM_GROUPS
            if i == 1
                Ly = Y(find(diagnosis_grouping == i),:) * U(i : i * size(Y,2),:);
            else
                Ly = [Ly ;Y(find(diagnosis_grouping == i),:) * U(1 + size(Y,2)*(i - 1) : i * size(Y,2),:)];
            end
        end
    otherwise
        error('ï¿½ï¿½Ïµï¿½ï¿½ï¿½ï¿½Rï¿½Ä¼ï¿½ï¿½ã·½Ê½ï¿½ï¿½Ê±Ö»ï¿½ï¿½Îª1ï¿½ï¿½2')
end

%% 5. Permutation testing for LV significance

clear perm_order Xp Yp Rp Up Sp Vp rotatemat permsamp sp mypvals mySignifLVs numSignifLVs

disp('... Permutations ...')
for iter_perm = 1: NUM_PERMS
    
    % Display number of permutations (every 50 permuts)
    if mod(iter_perm,50) == 0, disp(num2str(iter_perm)); end
    
    % Leave X unchanged (no need to permute both X and Y matrices)
    Xp = X; % X is already normalized
    
    % Permute Y by shuffling rows (subjects) within groups
    perm_order = PermuteSort(diagnosis_grouping,Time); 
    Yp = Y0(perm_order,:);

    
    % Normalize permuted Y
    Yp = myPLS_norm(Yp,NUM_GROUPS,diagnosis_grouping,CONST_NORM);
    
    % Cross-covariance matrix between X and permuted Y
    Rp = myPLS_cov(Xp,Yp,CONST_NORM,diagnosis_grouping);
    
    % SVD of Rp
    [Up,Sp,Vp] = svd(Rp,'econ');
    
    % Procrustas transform (to correct for axis rotation/reflection)
    rotatemat = rri_bootprocrust(U, Up);
    Up = Up * Sp * rotatemat; 
    Sp = sqrt(sum(Up.^2)); 
    
    % Keep singular values for sample distribution of singular values
%     Sp = diag(Sp')';
    permsamp(:,iter_perm) = Sp';
    
    if iter_perm == 1, 
        sp = (Sp' >= diag(S));
    else
        sp = sp + (Sp' >= diag(S));
    end
    
end

myLVpvals = (sp + 1) ./ (NUM_PERMS + 1);
mySignifLVs = find(myLVpvals<0.05); % index of significant LVs
numSignifLVs = size(mySignifLVs,1); % number of significant LVs

% Display significant LVs
disp([num2str(numSignifLVs) 'significant LV(s)']);
for iter_lv = 1:numSignifLVs
    this_lv = mySignifLVs(iter_lv);
    disp(['LV' num2str(this_lv) ' - p=' num2str(myLVpvals(this_lv),'%0.3f') ]);
end

%% 6. Bootstrapping to test stability of brain saliences

clear all_boot_orders Xb Yb Rb Ub Sb Vb rotatemat Vbmean Vbmean2 Ubmean Ubmean2 Ub_std Vb_std Ures Vres


all_boot_orders = bootstrap_order(diagnosis_grouping,Time,NUM_BOOTSTRAP);   
clear i group_num
%%
disp('... Bootstrapping ...');
for iter_boot = 1 : NUM_BOOTSTRAP
    
    % Display number of bootstraps (every 50 samples)
    if mod(iter_boot,50) == 0, disp(num2str(iter_boot)); end
    
    % Bootstrap of X
    Xb =X0(all_boot_orders(:,iter_boot),:);
    Xb = myPLS_norm(Xb,NUM_GROUPS,diagnosis_grouping,CONST_NORM);
    
    % Bootstrap of Y
    Yb = Y0(all_boot_orders(:,iter_boot),:);
    Yb = myPLS_norm(Yb,NUM_GROUPS,diagnosis_grouping,CONST_NORM);
    
    % Bootstrap version of R
    Rb = myPLS_cov(Xb,Yb,CONST_NORM,diagnosis_grouping);
    
    % SVD of Rb
    [Ub,Sb,Vb] = svd(Rb,'econ');
    
    % Procrustas transform (to correct for axis rotation/reflection)
    rotatemat1 = rri_bootprocrust(U, Ub);
    rotatemat2 = rri_bootprocrust(V, Vb);
    Vb = Vb * rotatemat1;
    Ub = Ub * rotatemat2;
    
    switch CONST_NORM
        case 1
            Lx_b = Xb * Vb; % Brain scores : original imaging data projected on brain saliences
            Ly_b = Yb * Ub; % Behavior scores : original behavior data projected on behavior saliences
        case 2
            Lx_b = Xb * Vb;
            for i = 1: NUM_GROUPS
                if i == 1
                    Ly_b = Yb(find(diagnosis_grouping == i),:) * Ub(i : i * size(Yb,2),:);
                else
                    Ly_b = [Ly_b ;Yb(find(diagnosis_grouping == i),:) * Ub(1 + size(Yb,2)*(i - 1) : i * size(Yb,2),:)];
                end
            end
        otherwise
            error('ï¿½ï¿½Ïµï¿½ï¿½ï¿½ï¿½Rï¿½Ä¼ï¿½ï¿½ã·½Ê½ï¿½ï¿½Ê±Ö»ï¿½ï¿½Îª1ï¿½ï¿½2')
    end
    % Online computing of mean and variance
    if iter_boot == 1,
        Vb_mean = Vb;
        Ub_mean = Ub;
        Vb_mean2 = Vb.^2;
        Ub_mean2 = Ub.^2;
    else
        Vb_mean = Vb_mean + Vb;
        Ub_mean = Ub_mean + Ub;
        Vb_mean2 = Vb_mean2 + Vb.^2;
        Ub_mean2 = Ub_mean2 + Ub.^2;
    end
    clear Lx_b Ly_b r1 r2
end
% Calculation of standard errors of saliences
Ub_mean = Ub_mean / NUM_BOOTSTRAP;
Ub_mean2 = Ub_mean2 / NUM_BOOTSTRAP;
Vb_mean = Vb_mean / NUM_BOOTSTRAP;
Vb_mean2 = Vb_mean2 / NUM_BOOTSTRAP;

Ub_std = sqrt(Ub_mean2 - Ub_mean.^2); Ub_std = real(Ub_std);    % realÈ¡Êµï¿½ï¿½
Vb_std = sqrt(Vb_mean2 - Vb_mean.^2); 
 
% Bootstrap ratio (ratio of saliences by standard error)
Ures = U ./ Ub_std;
Vres = V ./ Vb_std; 

% Change bootstrapped saliences in case they contain infinity 
% (when Ub_std/Vb_std are close to 0)
inf_Vvals = find(~isfinite(Vres));  % isfiniteï¿½ï¿½ï¿½Þ·ï¿½ï¿½ï¿½1ï¿½ï¿½ï¿½ï¿½ï¿½Þ£ï¿½nanï¿½ï¿½ï¿½ï¿½ï¿½ï¿½0ï¿½ï¿½
for iter_inf = 1: size(inf_Vvals,1)
    Vres(inf_Vvals(iter_inf)) = V(inf_Vvals(iter_inf));
end
inf_Uvals = find(~isfinite(Ures));
for iter_inf = 1: size(inf_Uvals,1)
    Ures(inf_Uvals(iter_inf)) = U(inf_Uvals(iter_inf));
end

clear inf_Vvals inf_Uvals iter_inf

%% Contribution of original variables to LVs
% Brain & behavior structure coefficients (Correlations imaging/behavior variables - brain/behavior scores)

clear myBrainStructCoeff myBehavStructCoeff

% Brain structure coefficients
for iter_lv = 1:numSignifLVs
    this_lv = mySignifLVs(iter_lv);
    
    for iter_img = 1:size(X,2)
        clear tmpy tmpx r p l
        switch CONST_NORM
            case 1
                tmpx = X(:,iter_img);
                tmpy = Ly(:,this_lv);
                [r,p] = corrcoef(tmpx,tmpy.');
                myBrainStructCoeff(iter_img,iter_lv) = r(1,2);
            case 2
                for i = 1: NUM_GROUPS
                    l = find(diagnosis_grouping == i);
                    tmpx = X(l,iter_img);
                    tmpy = Ly(l,this_lv);
                    [r,p] = corrcoef(tmpx,tmpy.');
                    myBrainStructCoeff(iter_img,iter_lv,i) = r(1,2);
                    myBrainStructCoeff_p(iter_img,iter_lv,i) = p(1,2);
                end
        end
    end
    
end
% Behavior structure coefficients
for iter_lv = 1: numSignifLVs
    this_lv = mySignifLVs(iter_lv);
    
    for iter_behav = 1:size(Y,2),
        clear tmpy tmpx r p
        tmpx = Y(:,iter_behav);
        tmpy = Lx(:,this_lv);        
        [r,p] = corrcoef(tmpx,tmpy.');
        myBehavStructCoeff(iter_behav,iter_lv) = r(1,2);
    end
end



%% IV. Behavior saliences
for iter_lv = 1: numSignifLVs
    this_lv = mySignifLVs(iter_lv);
    
   figure('Visible','off');
    switch CONST_NORM
        case 1
            bar(reshape(U(:,this_lv),[CONST_NUM_BEHAV 1]),0.5,'b');
            axis([0 size(Y,2)+1 min(U(:,this_lv))-0.1 max(U(:,this_lv))+0.2])
            hold on 
            for i = 1 : length(U(:,this_lv))
                if abs(Ures(i,this_lv)) > BSRthreshold
                    if Ures(i,this_lv) >= 0
                        plot(i,U(i,this_lv)+0.02,'k*')
                    else
                        plot(i,U(i,this_lv)-0.02,'k*')
                    end
                end
                hold on
            end
        case 2
            U_med = U(:,this_lv);
            for i = 1: CONST_NUM_BEHAV
                for j = 1: NUM_GROUPS
                    U_fig(i,j) = U_med(i + (j-1) * CONST_NUM_BEHAV);
                end
            end
            %% ï¿½ï¿½Ê±ï¿½ï¿½ï¿½ò£¬½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿?
            x1 = 0.85:1:size(U_fig,1)-0.15;
            x2 = 1.15:1:size(U_fig,1)+0.15;
            bar(x1,U_fig(:,1),0.3,'FaceColor',[255 69 0]/255)
            hold on
            bar(x2,U_fig(:,2),0.3,'FaceColor',[30 144 255]/255)
            axis([0 size(U_fig,1)+1 min(U_fig(:))-0.1 max(U_fig(:))+0.2])
            hold on
            for i = 1 : size(U_fig,1)
                if abs(Ures(i,this_lv)) > BSRthreshold
                    if U_fig(i,1) > 0
                        plot(x1(i),U_fig(i,1)+0.02,'*','color',[255 69 0]/255)
                    else
                        plot(x1(i),U_fig(i,1)-0.02,'*','color',[255 69 0]/255)
                    end
                end
            end
            hold on
            for i = 1 : size(U_fig,1)
                if abs(Ures(i+size(U_fig,1),this_lv)) > BSRthreshold
                    if U_fig(i,2) > 0
                        plot(x2(i),U_fig(i,2)+0.02,'*','color',[30 144 255]/255)
                    else
                        plot(x2(i),U_fig(i,2)-0.02,'*','color',[30 144 255]/255)
                    end
                end
            end
            %%
            legend(CONST_DIAGNOSIS,'FontSize',8)
%             legend(CONST_DIAGNOSIS,12)
            clear U_med U_fig i j x1 x2
        otherwise
            disp('Î´ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½È«ï¿½ï¿½ï¿½ï¿½Ê±ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½1ï¿½ï¿½2')
            error('ï¿½ï¿½È¤...') 
    end
    hold on
    set(gca,'XTickLabel',CONST_BEHAV_NAMES,'FontSize',8)
    xlabel('Behavioral variables','FontSize',12);
    ylabel('Behavioral saliences','FontSize',12);
    set(gca,'xtick',1:length(CONST_BEHAV_NAMES),'xtickLabel',CONST_BEHAV_NAMES)
    title({['LV' num2str(this_lv) ' - ' num2str(100*explVarLVs(this_lv),'%.2f') '% of covariance'],['Behavior saliences']},'FontSize',12);
    hold on
    hold off
    saveas(gcf,fullfile(outputPath,['LV' num2str(this_lv) '_BrainSalience']),'fig');
end

%% V. Behavior structure coefficients
% Display top values only 

numTopVals = CONST_NUM_BEHAV; % number of top correlations to display
if CONST_NUM_BEHAV > 20 
    numTopVals = 20;        
end

for iter_lv = 1: numSignifLVs
    this_lv = mySignifLVs(iter_lv);

    clear absCorrs sortedCorrs sortIdx sortedCorrs2 sortIdx2 top_absCorrs top_vars 
    
    % Sort absolute values of correlations and select top values
    absCorrs = abs(myBehavStructCoeff(:,iter_lv));
    [sortedCorrs,sortIdx] = sort(absCorrs,'descend');
    top_absCorrs = myBehavStructCoeff(sortIdx(1:numTopVals),iter_lv);
    top_vars = CONST_BEHAV_NAMES(sortIdx(1:numTopVals))';
    
    % Sort again within the top values
    [sortedCorrs2,sortIdx2] = sort(top_absCorrs,'descend');
    mySortedTopCorrs(:,iter_lv) = top_absCorrs(sortIdx2);
    mySortedTopVars{:,iter_lv} = top_vars(sortIdx2);
    
    figure('Visible','off');
    bar(mySortedTopCorrs(:,iter_lv),0.5);
    set(gca,'XTickLabel',mySortedTopVars{iter_lv},'FontSize',8)
    xlabel('Behavioral variables','FontSize',12);
    ylabel('Correlations','FontSize',12);
    title(['LV' num2str(this_lv) ' - Behavior structure coefficients'],'FontSize',12);
    saveas(gcf,fullfile(outputPath,['LV' num2str(this_lv) '_behavStructCoeff.png']),'png');
    
    disp(['Top correlations for LV' num2str(this_lv) ' :']);
    for iter_corr = 1:numTopVals
        disp([mySortedTopVars{iter_lv}{iter_corr} ' - r=' num2str(mySortedTopCorrs(iter_corr,iter_lv),'%0.2f')]);
    end
end

% Clear all temporary variables
clear num_subj_group Rb Vp Up Vb Ub Xp Yp Yb Xb perm_order rotatemat Rp Sb Sp thisY0 thisYp low_CI high_CI tmpx tmpy r p iter_perm iter_lv this_lv iter_img iter_lv idx iter_behav iter_boot iter_group 

%% Save workspace with results
clear A absCorrs Ai boot_order i iter_corr myBSR myMap myMax myMinv r_BehavScore Vb_mean2 Vb_mean Vb_std
save(fullfile(outputPath,[resultsFilename '.mat']));
close all 
end