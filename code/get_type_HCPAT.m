function [type_name,type_Data]=get_type_HCPAT(features,behaver,behaver_name,numType,clusterWay,PLSCoutputPath)
%% plsc analysis and clustering
%  features: Rows correspond to subjects, columns correspond to feature variables.
%  behavior: Rows correspond to subjects, columns correspond to behavior variables.
%  behavior_name: The name of each behavior.
%  numType£ºThe number of subtypes needed.
%  clusterWay£ºThe way of clustring.1£ºk-means clustering;2:hierarchical clustering.
%  PLSCoutputPath: The path for saving plsc output.
diagnosis_grouping=ones(size(features,1),1);diagnosis{1}='patients';
[mySignifLVs,Lx,V]=myPLSC(PLSC_features,behaver,behaver_name,diagnosis_grouping,diagnosis,PLSCoutputPath);
type_Data.V=V(:,mySignifLVs);

% clustering based the result of plsc
Cluster_component=Lx(:,mySignifLVs);
clusterResults_Ind=zeros(size(features,1),1);
if clusterWay==1
[clusterResults_Ind,C]=kmeans(Cluster_component,numType,'Distance','cityblock');
end
if clusterWay==2
    Y=pdist(Cluster_component,'euclidean');  
    Z=linkage(Y,'ward'); 
    clusterResults_Ind=cluster(Z,numType);
    num=unique(clusterResults_Ind);
    C=zeros(length(num),size(Cluster_component,2));
    for ind=1:length(num)
        C(ind,:)=mean(Cluster_component(find(clusterResults_Ind==num(ind)),:));
    end
        
end

% find each type and save
type_name=unique(clusterResults_Ind);
type_ind=cell(length(type_name),1);
for i=1:length(type_name)
type_ind{i,1}=find(clusterResults_Ind==type_name(i));
type_Data.feature{i}=features(type_ind{i,1},:);
end
type_Data.typeInd=clusterResults_Ind;
type_Data.Cluster_component=Cluster_component;
type_Data.centroid=C;
end
