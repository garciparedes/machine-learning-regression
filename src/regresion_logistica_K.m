#
# Name: Logistic Regresion
# Subject: Tecnicas de Aprendizaje Automatico
# Author: Calonge Cano, Teodoro
# Author: García Prado, Sergio
# Author: Fernández Angulo, Oscar

function w = regresion_logistica_K(x, y)
  w = zeros( size(x,2),1);
  nu = 1 ./ (1 .+ e.^-(x * w));
  s = (nu .* (1 .- nu));
  w = inv(x' .* s' * x)*x'*(s'*x*w + y - nu);
end
