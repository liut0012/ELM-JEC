clear
addpath(genpath('functions'));

load iris;
filename = ['result_elmjec_iris'];
nclass = length(unique(gnd));

options.GraphDistanceFunction='euclidean';
options.GraphWeights='binary';
options.LaplacianNormalize=0;
options.LaplacianDegree=1;



[nFea,nSmp] = size(data);
num_class = length(unique(gnd));

data = mapminmax(data,-1,1); 
options.NN = 5;
L=laplacian(options,data');

acc = [];
for repeat = 1:10
    [G_all,Q_all,F_all] = elmjec(data,L,4, num_class,4,8,1000);
    for i = 1:size(G_all, 1)
        g = G_all(i,:);
        y(i) = find(g);
    end
    
    acc(repeat) = accuracy(gnd, y')
end

acc_mean = mean(acc)
acc_std = std(acc)
acc_max = max(acc)

result= [acc_mean, acc_std, acc_max];

save(filename);