#
# Name: Multiple Linear Regresion
# Subject: Tecnicas de Aprendizaje Automatico
# Author: Teodoro Calonge Cano

function w = regresion_lineal_K(x, y)
  X = ones(size(x)(1), size(x)(2)+1);
  X(:, 1:size(x)(2)) = x;
  A = zeros(size(x)(2)+1, size(x)(2)+1);
  B = zeros(size(x)(2)+1, 1);
  for i=1:size(x)(2)+1
    for j=1:size(x)(2)+1
      A(i,j)=sum(X(:,i).*X(:,j));
    endfor
    B(i,1)=sum(y.*X(:,i));
  endfor
  w=inv(A)*B;
endfunction
