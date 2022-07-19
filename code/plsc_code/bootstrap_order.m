function Result = bootstrap_order(diagnosis_grouping,Time,NUM_BOOTSTRAP)
    NumGroup = unique(diagnosis_grouping);
    for i = 1: length(NumGroup)
        GroupNum(i) = length(find(diagnosis_grouping == NumGroup(i))) / Time;
    end
    Result = [];
    for n = 1: NUM_BOOTSTRAP
        clear MidResult
        Result1 = [];
        for i = 1: length(NumGroup)
            MidResult{i} = ceil(GroupNum(i)*rand(GroupNum(i),1)) + (sum(GroupNum(1:i-1)) * (i-1) * Time);
            for k = 1: Time
                Result1 = [Result1;MidResult{i} + (k-1)*GroupNum(i)];
            end
        end
        Result = [Result,Result1];
    end
end