
clc

%two hidden layers of 16 nerons each INPUT layer of 784 28x28 pixcel photo
 %output 10 0-9 number
 %there weights are in matrix of W1,W2,W3
 %intial weights asign 
 W1=randn(16,784);
 W2=randn(16,16);
 W3=randn(10,16);
 B1=randn(16,1);
 B2=randn(16,1);
 B3=randn(10,1);
 
 Input=importdata('mnist_train.csv');
 TestInput=importdata('mnist_test.csv');
 for e=1:30
 Input=Input(randperm(size(Input,1)),:);
 y=Input(:,1);
 TestInput=TestInput(randperm(size(TestInput,1)),:);
 Test_y=TestInput(:,1);
 Test_input=(1/256)*TestInput(:,2:785);
 input=(1/256)*Input(:,2:785); 

 %training the data
 for j=1:3000
     
 n=1.5; %rate of the change in weights
 epoch=randi(60000,1,10);
 sum_dcw3=zeros(10,16);
 sum_dcb3=zeros(10,1);
 sum_dcw2=zeros(16,16);
 sum_dcb2=zeros(16,1);
 sum_dcw1=zeros(16,784);
 sum_dcb1=zeros(16,1);
 sum_cost=0;
 for i=1:10
     Y=zeros(10,1);
     val=y(epoch(i))+1;
 I=input(epoch(i),:)';
 Y(val)=1;
 % hidden layers
 Z1=W1*I+B1;
 A1=sig(Z1);
 Z2=W2*A1+B2;
 A2=sig(Z2);
 Z3=W3*A2+B3;
 A3=sig(Z3);
 
 %cost function
 
 C=sum((A3-Y).^2);
 
%change in the weights for layer 3
dc=2*(A3-Y).*dsig(Z3);
dcw3=dc*A2';
dcb3=dc;
%change in the weights for layer 2
dca2=W3'*dcb3;
dcb2=dca2.*dsig(Z2);
dcw2=dcb2*A1';
%change in weights for layer 1
dca1=W2'*dcb2;
dcb1=dca1.*dsig(Z1);
dcw1=dcb1*I';

%sum the gradients of Cost
 sum_dcw3=sum_dcw3+dcw3;
 sum_dcb3=sum_dcb3+dcb3;
 sum_dcw2=sum_dcw2+dcw2;
 sum_dcb2=sum_dcb2+dcb2;
 sum_dcw1=sum_dcw1+dcw1;
 sum_dcb1=sum_dcb1+dcb1;
 sum_cost=sum_cost+C;

 end

 %altering the weights
 W3=W3-(n/10)*sum_dcw3;
 B3=B3-(n/10)*sum_dcb3;
 W2=W2-(n/10)*sum_dcw2;
 B2=B2-(n/10)*sum_dcb2;
 W1=W1-(n/10)*sum_dcw1;
 B1=B1-(n/10)*sum_dcb1;
 sum_cost/10;


 

 end
 r=0;
% testing the data 
 for i=1:10000
   I=Test_input(i,:)';
   Ans=Test_y(i);
  Z1=W1*I+B1;
 A1=sig(Z1);
 Z2=W2*A1+B2;
 A2=sig(Z2);
 Z3=W3*A2+B3;
 A3=sig(Z3);
 [v,l]=max(A3);
 if l-1==Ans
     r=r+1;
 end
 
 end
 r
 end
 