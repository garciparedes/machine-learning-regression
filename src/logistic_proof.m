#
# Name: Logistic Regresion
# DataSet: example
# Subject: Tecnicas de Aprendizaje Automatico
# Author: Calonge Cano, Teodoro
# Author: García Prado, Sergio
# Author: Fernández Angulo, Oscar


# cargar datos. 'wine_data.csv' no lleva nombres de campos.

dataset=csvread('./../datasets/example.csv', 1, 0);

# Aislar x e y

x=dataset(:,1:size(dataset)(2)-1);
y=dataset(:,size(dataset)(2));

# Construir los conjuntos de entrenamiento
x_ampliado = ones(size(x)(1), size(x)(2)+1);
x_ampliado(:, 1)=ones(size(x_ampliado)(1), 1);
x_ampliado(:, 2:size(x)(2)+1) = x;

# Regresion Lineal

w = regresion_logistica_K(x_ampliado, y);
1 ./ (1 + e.^(x_ampliado * w))
logistic_rate = sum(1 ./ (1 + e.^(x_ampliado * w)) < 0.5 == y,1) /size(y,1)
