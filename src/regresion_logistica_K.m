#
# Name: Multiple Linear Regresion
# Subject: Tecnicas de Aprendizaje Automatico
# Author: Teodoro Calonge Cano

function w = regresion_logistica_K(x, y, alpha=0.3, epocs=10)
  #x = 1.0 ./ ( 1.0 + exp(-(x)./norm(x)));
  #w = regresion_lineal_K(x,y);
  #w = w./norm(w);
  w = zeros(size(x)(2), 1);

  for t=1:epocs
    for i=1:size(x)(1)
      p = 1 / (1 + e^(-(x(i,:) * w)));
      w = w + alpha * (y(i) - p) * p * (1 - p) * x(i,:)';
    endfor
  endfor
endfunction
