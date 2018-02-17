%Use SGD() function and input the parameters in the function: 
%a) nnsize: the vector of the nodelayers 
%b) traning_data and training_target: input set and target set  
%c) epochs 
%d)  mini_batch_size 
%e) eta 
%f) test_data, test_target: here I don’t split the data into training and test set, so for this parameters I just input the same data set as the training_data and training_target 


function SGD(nnsize,training_x,training_y,epochs,batchSize,eta)
global layerNum
global weights
global biases
global n
layerNum=length(nnsize);
n=length(training_x);
 
%initiate random weights and biases with mean=0 and std=1
weights={};
biases={};
for p=1:layerNum-1;
    weights{p}=randn(nnsize(p+1),nnsize(p));
    biases{p}=randn(nnsize(p+1),1);
end
 
%SGD
for i=1:epochs;
    %shuffle data
    id = randperm(length(training_x));
    training_x=training_x(:,id);
    training_y=training_y(:,id);
    %update mini_batch and get the new weights and biases
    update_mini_batch(training_x,training_y,batchSize,eta);
    %evaluate the network with test data
    [MSE,correct,m]=evaluate(training_x,training_y);
    if correct==m
        %print the result
        fprintf('Epoch %g,MSE:%g,Correct:%g / %g,Acc:%g\n',i,MSE,correct,m,correct/m)
    else
        fprintf('Epoch %g,MSE:%g,Correct:%g / %g,Acc:%g\n',i,MSE,correct,m,correct/m)
    
    end
end


function update_mini_batch(x,y,batchSize,eta)
global weights
global biases
global layerNum
global n
gn=(0:floor(n/batchSize));
for k=1:length(gn)-1;
    if k==1 
        batch_x=x(:,(1:gn(k+1)*batchSize));
        batch_y=y(:,(1:gn(k+1)*batchSize));
    else
        batch_x=x(:,(gn(k)*batchSize+1:gn(k+1)*batchSize));
        batch_y=y(:,(gn(k)*batchSize+1:gn(k+1)*batchSize));
    end
    %get the nabla_b and nabla_b
    [dnw, dnb]=backprop(batch_x,batch_y);
    %calculate updated weights and biases
    for i=1:layerNum-1
        weights{i}=weights{i}-(eta/batchSize)*dnw{i};
        biases{i}=biases{i}-(eta/batchSize)*sum(dnb{i},2);
    end
end
end


function [nabla_w,nabla_b]=backprop(x,y)
global weights
global biases
global layerNum
nabla_b={};
nabla_w={};
for i = 1:layerNum-1     
    nabla_w{i}=zeros(size(weights{i}));     
    nabla_b{i}=zeros(size(biases{i}));
end
%feedforward
activation=x;
activations={x};
zs={};
for i=1:layerNum-1;
    z=bsxfun(@plus,weights{i}*activation,biases{i});
    zs{i}=z;
    activation=sigmoid(z);
    activations{i+1}=activation;
end
%backward pass
delta=(activations{end}-y).*sigmoid_prime(zs{end});
nabla_b{end}=delta;
nabla_w{end}=delta*activations{end-1}';
for j=(layerNum-1):-1:2
    z=zs{j-1};
    sp=sigmoid_prime(z);
    delta=(weights{j}')*delta.*sp;
    nabla_b{j-1}=delta;
    nabla_w{j-1}=delta*(activations{j-1}');
end
end


function [MSE,correct,m]=evaluate(x,y)
m=size(y,2);
% create empty zeros vector to save predicted output
output_node=size(y,1);
if output_node>1;
    [argvalue_t,target]=max(y);
else
    target=y;
end
% Acquire predicted output by classifying the predicted result to the class/node that containing the maximal output value
res=feedforward(x);
MSE=sum((sum((res-y).^2,2)/output_node))/m;
% for XOR data set
if output_node==1
    pred=round(res);
% for iris and MNIST data set
else
    [argvalue_p,pred]=max(res);
end
correct=sum(pred==target);
end
 
function x=feedforward(x)
global weights
global biases
global layerNum
for i=1:layerNum-1
    x=sigmoid(bsxfun(@plus,weights{i}*x,biases{i}));
end
end


function res=sigmoid(x)
res=1./((1+exp(-x)));


function res=sigmoid(x)
res=1./((1+exp(-x)));
