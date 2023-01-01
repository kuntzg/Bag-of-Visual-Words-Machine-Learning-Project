function [w,b] = train_svm_primal(X,y,lambda)
[m,d]=size(X);

H=zeros(d+m,d+m);
for i=1:d
    H(i,i)=2*lambda;
end

f=zeros(1,d+m);
f(d+1:end)=1/m;
A=[X.*repmat(-y,[1,d]) -eye(m)];
A=[A;zeros(m,d) -eye(m)];
b2=zeros(2*m,1);
b2(1:m)=-1;
[sol]=quadprog(H,f,A,b2);

w=sol(1:d);

end

