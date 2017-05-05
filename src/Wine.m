#
# Name: Multiple Linear Regresion with Relative Errors
# DataSet: housing
# Subject: Tecnicas de Aprendizaje Automatico
# Author: Teodoro Calonge Cano
# Adapted by: Sergio García Prado


# cargar datos. 'wine_data.csv' no lleva nombres de campos.

dataset=csvread('./../datasets/wine.csv', 1, 0);

# Aislar x e y

x=dataset(:,1:size(dataset)(2)-1);

x_ampliado = ones(size(x)(1), size(x)(2)+1);
x_ampliado(:, 1)=ones(size(x_ampliado)(1), 1);
x_ampliado(:, 2:size(x)(2)+1) = x;

y=dataset(:,size(dataset)(2));

# Construir los conjuntos de entrenamiento

aleatorio=rand(size(dataset)(1),1);
hold_out_train=round(size(dataset)(1)*2/3);
hold_out_test=round(size(dataset)(1)/3);

x_index=resize(x, size(x)(1), size(x)(2)+1);
x_index(:,size(dataset)(2))=aleatorio;

x_sort=sortrows(x_index, size(dataset)(2));

x_train=x_sort(1:hold_out_train, 1:size(x)(2));

x_train_ampliado = ones(size(x_train)(1), size(x_train)(2)+1);
x_train_ampliado(:, 1)=ones(size(x_train)(1), 1);
x_train_ampliado(:, 2:size(x_train)(2)+1) = x_train;


y_index=resize(y, size(y)(1), size(y)(2)+1);
y_index(:,2)=aleatorio;

y_sort=sortrows(y_index, 2);

y_train=y_sort(1:hold_out_train, 1:size(y)(2));

# Construir los conjuntos de test

x_test=x_sort((hold_out_train+1):size(x)(1), 1:size(x)(2));


x_test_ampliado = ones(size(x_test)(1), size(x_test)(2)+1);
x_test_ampliado(:, 1)=ones(size(x_test)(1), 1);
x_test_ampliado(:, 2:size(x_test)(2)+1) = x_test;


y_test=y_sort((hold_out_train+1):size(y)(1), 1:size(y)(2));

# Regresion Lineal

w=regresion_lineal_K(x_train_ampliado, y_train);


# Salida REAL Test

y_p=x_test_ampliado*w;
data_train=zeros(size(x_train)(1), size(x_train)(2)+1);
data_train(1:size(data_train)(1), 1:(size(data_train)(2)-1))=x_train;
data_train(1:size(data_train)(1), size(data_train)(2))=y_train;

#Error
error_absoluto=abs(y_test - y_p);
error_relativo=abs((y_test - y_p) ./ y_test).*100;

# Tasa de Aciertos
tasa_acierto_10=100*sum(error_relativo <= 10.0)/hold_out_test;
tasa_acierto_15=100*sum(error_relativo <= 15.0)/hold_out_test;
tasa_acierto_20=100*sum(error_relativo <= 20.0)/hold_out_test;
tasa_acierto_25=100*sum(error_relativo <= 25.0)/hold_out_test;

# Tasa de Error
tasa_fallo_10=100-tasa_acierto_10
tasa_fallo_15=100-tasa_acierto_15
tasa_fallo_20=100-tasa_acierto_20
tasa_fallo_25=100-tasa_acierto_25


y_m = y_train == unique(y_train)';

c_size = (size(y_train,2)^2-size(y_train,2))/2;
w_m = zeros(size(x_ampliado,2), c_size);

ii = 0;
for i = 1:c_size-1
  for j = i+1:c_size
    ii += 1;
    c_index = (y_m(:,i) + y_m(:,j)) == 1;
    y_bin = y_train(c_index,:) == unique(y_train(c_index))(2);
    x_bin = x_train_ampliado(c_index,:);
    w_m(:,ii) = regresion_logistica_K(x_bin, y_bin);
  end
end
c_out = 1 ./ (1 + e.^(x_test_ampliado * w_m)) < 0.5;
c_divided = zeros(size(c_out,1),c_size);

ii = 0;
for i = 1:c_size-1
  for j = i+1:c_size
    ii += 1;
    for k = 1:size(c_out,1)
      if (c_out(k,ii) == 1)
        c_divided(k,j) += 1;
      else
        c_divided(k,i) += 1;
      end
    end
  end
end
correct = 0;
for i = 1:size(c_divided,1)
  [M,I] = max(c_divided(i,:))(2);
  correct += (I == y_test(i));
end
tasa_acierto_logistic = 100 * correct /size(y_test,1);
tasa_fallo_logistic = 100 -tasa_acierto_logistic

%{

# Formación de ficheros de entrenamiento y test
data_train=zeros(size(x_train)(1), size(x_train)(2)+1);
data_train(1:size(data_train)(1), 1:(size(data_train)(2)-1))=x_train;
data_train(1:size(data_train)(1), size(data_train)(2))=y_train;
csvwrite('data_wine_train_octave.csv', data_train);

data_test=zeros(size(x_test)(1), size(x_test)(2)+1);
data_test(1:size(data_test)(1), 1:(size(data_test)(2)-1))=x_test;
data_test(1:size(data_test)(1), size(data_test)(2))=y_test;
csvwrite('data_wine_test_octave.csv', data_test);

%}
