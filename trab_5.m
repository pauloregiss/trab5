clear,clc
ordem = input('Digite a ordem do preditor: ');
N = input('Digite a quantidade de neurônios na camada escondida: ');

for T = 1:3 % Serão realizados 3 treinamentos
    p = ordem;
    % Obter o conjunto de amostras de treinamento
    dados = load('dados_de_treinamento.dat');
    qtde_dados_treinamento = size(dados,1);

    % Inicializar as matrizes pesos W1 (Neuronios Intermediário x Entradas) e
    % W2(Neurônios saida x Neuronios Intermediários+1) aleatoriamente com
    % valores aleatórios pequenos
    neuronios_intermediarios = N; %N
    neuronios_saida = 1;
    entradas = p+1;

    w1 = random('Uniform',0,1,neuronios_intermediarios,entradas);
    w2 = random('Uniform',0,1,neuronios_saida,neuronios_intermediarios+1);

    % Taxa de aprendizagem (ta) e precisão
    ta        = 0.1;
    precisao  = 0.5*10^-6;

    % Iniciar o contador de épocas
    ep = 1;

    % Iniciar o Erro Quadrático Médio atual
    EQM = 0;

    % Laço principal
    while true
        for k = 1:qtde_dados_treinamento - p

            x = fliplr(dados(k:p+k-1)');
            x = [-1 x];
            d = dados(p+k);

         % Fase Foward
         % Nx1   Nx(p+1) (p+1)x1  Nx1              Nx1  (N+1)x1
            I1  =  w1   *   x';   Y1 = 1./(1 + exp(-I1)); Y1 = [-1; Y1];
         % 1x1 1x(N+1)(N+1)x1 1x1               1x1
            I2  = w2 * Y1;    Y2 = 1./(1 + exp(-I2));

         % Fase backward
         %Derivada da função sigmóide em I1
         % Nx1      Nx1          Nx1
            a = exp(-I1)./(1+exp(-I1)).^2;
         %Derivada da função sigmóide em I2
         % 1x1      1x1          1x1
            b = exp(-I2)./(1+exp(-I2)).^2;
         %    1x1     1x1   1x1 1x1
            delta2   = b .* (d'-Y2);
         % 1x(N+1) 1x(N+1)   1x1    1x1   1x(N+1)
            w2   =   w2  +  (ta * delta2) * Y1';
         %    Nx1     Nx1    [1x1   1xN]'
            delta1   = a .* (delta2'*w2(:,2:neuronios_intermediarios+1))';
         %  Nx(p+1)  Nx(p+1)  1x1   Nx1   1x(p+1)
            w1   =    w1   +  ta * delta1 * x;
        end

        % Obter saída da rede ajustada
        for k = 1:qtde_dados_treinamento - p
         %  Nx1   Nx(p+1)  (p+1)x1  Nx1               Nx1  (N+1)x1
            I1  =   w1   *   x';    Y1 = 1./(1 + exp(-I1)); Y1 = [-1; Y1];
         %  1x1 1x(N+1) (N+1)x1 1x1               1x1
            I2 =  w2   *  Y1;    Y2 = 1./(1 + exp(-I2));
         %  1x1            1x1 1x1
            EQ(:,k) = 0.5*((d'-Y2).^2);
        end

        % Cálculo do EQM
      % 1x1   1x1       1x1 1x1
        EQM = EQM + sum(EQ)/k;
        ep = ep + 1;
      % 1x1         1x1
        eqm(:,ep) = EQM;
        EQM = 0;
      %          1x1          1x1        1x1
        if (abs(eqm(ep) - eqm(ep-1)) < precisao)
            break
        end

    end
    disp(strcat('Treinamento T',num2str(T),' finalizado!'))

    %% Validação
    % Obter o conjunto de dados de validação
    validacao = load('dados_de_validacao.dat');
    qtde_dados_validacao = size(validacao,1);
    dados = [dados; validacao];
    quantidade_total_dados = qtde_dados_treinamento+qtde_dados_validacao;
    figure(1),plot(dados(1:qtde_dados_treinamento),'*-'),hold on
    eixo_x = qtde_dados_treinamento+1:length(dados);
    eixo_y = dados(qtde_dados_treinamento+1:length(dados));
    plot(eixo_x,eixo_y,'*-r'),grid
    legend('Treinamento','Validação')
    xlabel('Amostras'),ylabel('Saída da rede'),title(strcat('Treinamento T',num2str(T)))

    for k = qtde_dados_treinamento+1:quantidade_total_dados

        xv = fliplr(dados(k-p:k-1)');
        xv = [-1 xv];
        yv = dados(k);
        % Fase Foward
    %  Nx1   Nx(p+1) (p+1)x1   Nx1              Nx1  (N+1)x1
        I1  =  w1  *  xv';     Y1 = 1./(1 + exp(-I1)); Y1 = [-1; Y1];
    % 1x20  1x11 11x20  1x20              1x20
        I2  = w2 * Y1;    Y2(k-qtde_dados_treinamento) = 1./(1 + exp(-I2));

    end

    % Gráfico da comparação entre a saída calculada pela RNA e a desejada
    figure(2),plot(Y2,'o-k'),hold on,plot(validacao,'*-r'),grid
    legend('Saída estimada','Saída desejada','Location','north')
    xlabel('Amostras'),ylabel('Saída da rede'),title(strcat('Treinamento T',num2str(T)))
    % Salvar o gráfico
    grafico = gca;
    saveas(grafico, strcat('T', num2str(T), ' - Saídas.jpg'));

    % Gráficos do EQM
    figure(3),plot(eqm(:,2:size(eqm,2))),grid
    xlabel('Épocas'),ylabel('EQM'),title(strcat('Treinamento T',num2str(T)))
    % Salvar o gráfico do Épocas x EQM
    grafico = gca;
   saveas(figure(3), strcat('T', num2str(T), ' - EQM.jpg'), 'jpeg');

    % Dados de saída
    EQM_final = eqm(ep)
    Epocas = ep
    Saida_da_rede = Y2'
    % Erro Relativo Médio percentual ((Calculado-Real)/Real)/quantidade
    ER = 100*((validacao - Y2')./validacao); % em percentual
    ERM = sum(ER)/qtde_dados_validacao % em percentual
    % Variância do Erro Relativo (Somatório[(Xi-Xméd)^2])/N
    Variancia = var(ER) % em percentual
    % Salvar os dados finais
    save(strcat('T',num2str(T),' - Dados de saída'),...
        'EQM_final','Epocas','Saida_da_rede','ERM','Variancia')
    close all
end

