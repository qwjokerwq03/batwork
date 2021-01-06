function [sig]=sig(x)
l=length(x);
sig=zeros(l,1);
for i=1:l
sig(i,1)=1/(1+exp(-x(i)));
end