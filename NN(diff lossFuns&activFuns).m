function [tr_a,tr_c,val_a,val_c,tes_a,tes_c,weights,biases]=SGD1(nnsize,w,b,input_x,input_y,NumEpochs,cost,transferFunction,batchSize,eta,lmbda,split,momentum,earlyStop)
%--------------------------------------------------------------------------
%%SGD1 input detail:
%%nnsize:    Neurons number in each layer, i.e.[2 3 2] means totally three
%            layers,input layer has 2 neurons, hidden layers has 3 and output
%            layer has 2
%            Format--1 row matrix 
%%w/b:       weights/biases assigned to the network
%            Format--cell
%%input_x/y: Whole data set, columns represent instances and rows represent
%            features
%            Format--matrix
%%NumEpochs: Number of epochs to run
%            Format--int
%%cost:      Cost Function, 1=Quadratic, 2=Cross-Entropy, 3=Log-likelihood
%            Format--int
%%batchSize: Size of the Mini Batch
%            Format--int
%%eta:       Learning rate
%            Format--float
%%lmbda:     L2 Regularization
%            Format--float
%%split:     Data set split, [80 20 20] means dividing data into 80% training set, 10% validation set and 10% test set
%            Format--1 row matrix
%%transferFunction: 1=Sigmoid,2=Softmax with sigmoid,3=ReLu,4=Tanh
%            Format--int
%%earlyStop: input early stop criteria generalization loss
   %            (Smallest error of validation set at time t to t- 4/Error of optimum at time t)-1
%--------------------------------------------------------------------------
%%setting up global variable so that they can be used at different workspace
%               layerNum(Number of layers in the network) 
%               weights/biases
%               n(Number of instances in the data set)
global layerNum
global weights
global biases
global n
layerNum=length(nnsize);
n=length(input_x);
%%Initiate epoch and accuracy of training data to 0
epoch=0;
train_acc=0;
%%Create empty matrix to take generated accuracy/cost from
% training,validation and test of each epoch
tr_a=[];
tr_c=[];
val_a=[];   
val_c=[];
tes_a=[];
tes_c=[];
 
%%Shuffle whole data set
id = randperm(length(input_x));
input_x=input_x(:,id);
input_y=input_y(:,id);
%%Split data into training, validation and test set
split=split/100;
input_tra_x=input_x(:,(1:floor(n*split(1))));
input_tra_y=input_y(:,(1:floor(n*split(1))));
input_val_x=input_x(:,(floor(n*split(1))+1:floor(n*(split(1)+split(2)))));
input_val_y=input_y(:,(floor(n*split(1))+1:floor(n*(split(1)+split(2)))));
input_tes_x=input_x(:,(floor(n*(split(1)+split(2)))+1:end));
input_tes_y=input_y(:,(floor(n*(split(1)+split(2)))+1:end));
 
%%Assign the inherited weights and biases to global variables or randomly
% initiate new weights and biaes with mean=0 and std 1/sqrt(number of input)
if isempty(w) && isempty(b)
    weights={};
    biases={};
    for p=1:layerNum-1;
        weights{p}=randn(nnsize(p+1),nnsize(p))/sqrt(nnsize(p));
        biases{p}=randn(nnsize(p+1),1);
    end
else
    weights=w;
    biases=b;
end
    
 
fprintf('     |           TRAIN             ||          VALIDATION         ||           TEST              \n')
fprintf('-------------------------------------------------------------------------------------------------\n')
fprintf('Ep   |  Cost  |   Corr    |  Acc   ||  Cost  |   Corr    |  Acc   ||  Cost  |   Corr    |  Acc   \n')
fprintf('-------------------------------------------------------------------------------------------------\n')
     
while (epoch<NumEpochs) && (train_acc<1)
%%Early Stop check 
% first find out the minimum cost till current epoch as optimal cost
% second find the minimum cost of the last five epochs as recent cost 
% rate=recent cost/optimal cost
    % if the rate larger than the given number e.g. 0.5, then
    % stop the training
    if epoch>10
    val_opt=min(val_c);
    rate=min(val_c(end-5:end))/val_opt-1;
        if rate>earlyStop
            return
        end
    end
    
    epoch=epoch+1;
    
    %%Create mini batch data for training and update weights and biases for
    % current epoch
    update_mini_batch(input_tra_x,input_tra_y,cost,batchSize,eta,lmbda,transferFunction,momentum);
    
    %%evaluate the network by calculating the cost and accuracy of training,
    % validation and test set
    [train_cost,train_acc,train_correct,train_m]=evaluate(input_tra_x,input_tra_y,cost,transferFunction);
    tr_a(epoch)=train_acc;
    tr_c(epoch)=train_cost;
        
    [val_cost,val_acc,val_correct,val_m]=evaluate(input_val_x,input_val_y,cost,transferFunction);
    val_a(epoch)=val_acc;
    val_c(epoch)=val_cost;
        
    [test_cost,test_acc,test_correct,test_m]=evaluate(input_tes_x,input_tes_y,cost,transferFunction);
    tes_a(epoch)=test_acc;
    tes_c(epoch)=test_cost;
    fprintf('%4.4g | %.4f | %6g/%6g | %.4f || %.4f | %6g/%6g | %.4f || %.4f | %6g/%6g | %.4f \n',epoch,train_cost,train_correct,train_m,train_acc,val_cost,val_correct,val_m,val_acc,test_cost,test_correct,test_m,test_acc)
 
end
end


function update_mini_batch(x,y,cost,batchSize,eta,lmbda,transferFunction,momentum)
global weights
global biases
global layerNum
x_len=length(x);
    
%%initiate velocity weights cell
v_w={};
for i = 1:layerNum-1
    v_w{i}=zeros(size(weights{i}));     
end
 
%%Calculate how many sets of mini batch we need and create index 
if mod(x_len,batchSize)==0
    gn=(1:floor(x_len/batchSize));
else
    gn=(1:ceil(x_len/batchSize));
end
 
%%Shuffle the order of index
id = randperm(length(gn));
gn=gn(id);
%%According to the index to extract a range of training data as mini batch
for k=1:length(gn)
    if gn(k)==min(gn) 
        batch_x=x(:,(1:gn(k)*batchSize));
        batch_y=y(:,(1:gn(k)*batchSize));
    elseif gn(k)==max(gn)
        batch_x=x(:,((gn(k)-1)*batchSize+1:end));
        batch_y=y(:,((gn(k)-1)*batchSize+1:end));
    else
        batch_x=x(:,((gn(k)-1)*batchSize+1:gn(k)*batchSize));
        batch_y=y(:,((gn(k)-1)*batchSize+1:gn(k)*batchSize));
    end
 
    %%Use backpropagation to get the nabla_b and nabla_b
    [dnw, dnb]=backprop(batch_x,batch_y,cost,transferFunction);
 
    %%Update weights and biases, use lmbda by using weights=weights-eta*v_w
    % and biases=biases-eta*v_d
    for i=1:layerNum-1
        %%update velocity
        v_w{i}=momentum*v_w{i}+dnw{i};
        weights{i}=(1-eta*(lmbda/x_len))*weights{i}-(eta/batchSize)*v_w{i};
        biases{i}=biases{i}-(eta/batchSize)*sum(dnb{i},2);
    end
end
end

function [nabla_w,nabla_b]=backprop(x,y,cost,transferFunction)
global weights
global biases
global layerNum
    
%%Create cell with zeros matrixes for nabla b and nabla w
nabla_b={};
nabla_w={};
for i = 1:layerNum-1     
    nabla_w{i}=zeros(size(weights{i}));     
    nabla_b{i}=zeros(size(biases{i}));
end
 
activation=x;
activations={x};
zs={};
if transferFunction==1
    %%Feedforward,by using sigmoid transfer function except for the last layer
    for i=1:layerNum-1;
        z=bsxfun(@plus,weights{i}*activation,biases{i});
        zs{i}=z;
        activation=sigmoid(z);
        activations{i+1}=activation;
    end
    
elseif transferFunction==2
    %%Feedforward,by using sigmoid transfer function except for the last layer
    for i=1:layerNum-2;
        z=bsxfun(@plus,weights{i}*activation,biases{i});
        zs{i}=z;
        activation=sigmoid(z);
        activations{i+1}=activation;
    end
    %%Use softmax transfer function for last layer
    s=bsxfun(@plus,weights{end}*activation,biases{end});
    zs{end+1}=s;
    act=zeros(size(y));
    for j=1:length(s)
        exp(s(:,j));
        sum(exp(s(:,j)));
        act(:,j)=exp(s(:,j))/sum(exp(s(:,j)));
    end
    activations{end+1}=act;
 
elseif transferFunction==3
    %%Feedforward,by using ReLu transfer function except for the last layer
    for i=1:layerNum-2;
        z=bsxfun(@plus,weights{i}*activation,biases{i});
        zs{i}=z;
        activation=max(0,z);
        activations{i+1}=activation;
    end
    %%Last layer use softmax transfer function
    s=bsxfun(@plus,weights{end}*activation,biases{end});
    zs{end+1}=s;
    act=zeros(size(y));
    for j=1:length(s)
        exp(s(:,j));
        sum(exp(s(:,j)));
        act(:,j)=exp(s(:,j))/sum(exp(s(:,j)));
    end
    activations{end+1}=act;
    
elseif transferFunction==4
    %%Feedforward,by using sigmoid transfer function except for the last layer
    for i=1:layerNum-1;
        z=bsxfun(@plus,weights{i}*activation,biases{i});
        zs{i}=z;
        activation=tanh(z);
        activations{i+1}=activation;
    end
end
    
%%Backward pass, calculate last layer's error(delta)
% update the last layer's nabla w and labla b 
if cost==1
    delta=(activations{end}-y).*sigmoid_prime(zs{end});
else
    delta=activations{end}-y;
end
nabla_b{end}=delta;
nabla_w{end}=delta*activations{end-1}';
 
%%calculate hidden layer's error(delta) by using derivative sigmoid
% and update the hidden layer's nabla w and labla b 
for j=(layerNum-1):-1:2
    z=zs{j-1};
    if transferFunction==3
        z(z<0)=0.1;
        z(z>=0)=1;
        sp=z;
    elseif transferFunction==4
        sp=(1-tanh(z).^2);
    else
        sp=sigmoid_prime(z);
    end
    delta=(weights{j}')*delta.*sp;
    nabla_b{j-1}=delta;
    nabla_w{j-1}=delta*(activations{j-1}');
end
end



function [costRes,acc,correct,m]=evaluate(x,y,cost,transferFunction)
%%When the input data is empty, simply return cost,accuracy,correct number and size of input data to 0
costRes=0;
if isempty(x)
    costRes=0;
    acc=0;
    correct=0;
    m=0;
    return
else
    %%m is how many instances in input data
    m=size(y,2);
    %%If there are more than 1 row in target input(y), extract the index of 
    % the row with largest values for each instance as the reference labels for prediction  
    output_node=size(y,1);
    if output_node>1;
        [argvalue_t,target]=max(y);
    else
        target=y;
    end
    %%Calculate the cost of prediction according to different cost function.
    res=feedforward(x,transferFunction);
    if cost==1
        costRes=QuadraticCost(res,y)/(2*m);
    elseif cost==2
        costRes=CrossEntropyCost(res,y)/m;
    elseif cost==3
        for i=1:length(target)
            costRes=costRes+(-log(res(target(i),i)));
        end
        costRes=costRes/m;  
    end
        
    %%If there is only one output node,round up the predicted value as
    % predicted label. If there are more than one nodes, extract the index
    % of the nodes with largest values as predicted label
    if output_node==1
        pred=round(res);
    else
        [argvalue_p,pred]=max(res);
    end
    
    %%Compare the predicted labels with the reference labels to get the number of correct
    % classified instances, calculate accuracy
end
correct=sum(pred==target);
acc=correct/m;
end
 
function x=feedforward(x,transferFunction)
global weights
global biases
global layerNum
if transferFunction==1
    for i=1:layerNum-1
        x=sigmoid(bsxfun(@plus,weights{i}*x,biases{i}));
    end
elseif transferFunction==2
    for i=1:layerNum-2
        x=sigmoid(bsxfun(@plus,weights{i}*x,biases{i}));
    end
    z=bsxfun(@plus,weights{end}*x,biases{end});
    act=zeros(size(z));
    for j=1:length(z)
        act(:,j)=exp(z(:,j))/sum(exp(z(:,j)));
    end
    x=act;
elseif transferFunction==3
    for i=1:layerNum-2
        x=max(0,bsxfun(@plus,weights{i}*x,biases{i}));
    end
    z=bsxfun(@plus,weights{end}*x,biases{end});
    act=zeros(size(z));
    for j=1:length(z)
        act(:,j)=exp(z(:,j))/sum(exp(z(:,j)));
    end
    x=act;
elseif transferFunction==4
    for i=1:layerNum-1
        x=tanh(bsxfun(@plus,weights{i}*x,biases{i}));
    end
end
end



function res=CrossEntropyCost(x,y)
res=sum(sum((-y.*log(x))-(1-y).*log(1-x)));
end

function res=QuadraticCost(x,y)
res=sum((sum((x-y).^2,2)));
end

function res=sigmoid(x)
res=1./((1+exp(-x)));
end
function res=sigmoid_prime(x)
res=sigmoid(x).*(1-sigmoid(x));
end

function visual_acc(tr_a,val_a,tes_a,epoch)
    x=1:epoch;
    plot(x,tr_a,'x-',x,val_a,'--',x,tes_a,'-')
    legend('training','validation','test')
    xlabel('Epochs')
    ylabel('Accuracy')
end

x=(1:30);
plot(x,tr_c,'x-',x,val_c,'--',x,tes_c,'-')
legend('training','validation','test')
xlabel('Epochs')
ylabel('Error')

function visual_cost(tr_c,val_c,tes_c,epoch)
    x=1:epoch;
    plot(x,tr_c,'x-',x,val_c,'--',x,tes_c,'-')
    legend('training','validation','test')
    xlabel('Epochs')
    ylabel('Error')
end
