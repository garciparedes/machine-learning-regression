#
# Name: Multiple Linear Regresion
# Subject: Tecnicas de Aprendizaje Automatico
# Author: Teodoro Calonge Cano

function w = regresion_logistica_K(x, y)
  w = zeros( size(x,2),1);
  nu = zeros(size(x,2),1);
  s = zeros(size(x,1),1);
  nu = 1 ./ (1 .+ e.^-(x * w));
  s = (nu .* (1 .- nu));
  w = inv(x' .* s' * x)*x'*(s'*x*w + y - nu);
  w
end
