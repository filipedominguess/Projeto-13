# Projeto 13 - Previsão de Demanda de Táxi em Aeroportos

![Previsão de Demanda de Táxi](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQXjPP7qZos40Z2qacE1zlj1mP2P6n_RgIQ-w&usqp=CAU)

Este documento serve como uma documentação abrangente para o projeto "Previsão de Demanda de Táxi em Aeroportos" da empresa Sweet Lift Taxi. O projeto envolve a coleta, análise e previsão de pedidos de táxi em aeroportos durante o horário de pico, a fim de atrair mais motoristas. O objetivo é construir um modelo de previsão de demanda que atenda a um limite máximo de 48 para a métrica REQM no conjunto de teste.

## Descrição do Projeto

A empresa Sweet Lift Taxi coletou dados históricos sobre pedidos de táxi nos aeroportos e deseja desenvolver um modelo que preveja a quantidade de pedidos de táxi para a próxima hora. A previsão é crucial para a empresa atrair mais motoristas durante os horários de pico, otimizando a disponibilidade de táxis. O projeto inclui as seguintes etapas:

1. **Coleta de Dados**: Os dados históricos sobre pedidos de táxi são armazenados no arquivo "taxi.csv". É essencial fazer o download e carregar esses dados para análise.

2. **Análise de Dados**: Uma análise dos dados é realizada para entender tendências, sazonalidades e características que possam influenciar a demanda. A decomposição sazonal dos dados é uma parte crítica deste processo.

3. **Treinamento de Modelos**: Diferentes modelos de regressão são treinados com o objetivo de prever a demanda de táxi. No projeto, são considerados modelos de regressão linear (Linear Regression) e o algoritmo CatBoost (CatBoostRegressor) com otimização de hiperparâmetros.

4. **Teste e Avaliação**: Os modelos treinados são testados usando uma amostra de teste. A métrica de avaliação principal é o REQM (Erro Quadrático Médio). É fundamental garantir que o REQM no conjunto de teste não seja superior a 48.

## Estrutura do Projeto

O projeto é estruturado em várias etapas distintas:

1. **Preparação e Análise Inicial**:
   - Os dados são carregados e tratados.
   - A análise inicial inclui a visualização dos dados e a decomposição sazonal para identificar tendências e sazonalidades.

2. **Engenharia de Recursos**:
   - Os dados são transformados em recursos relevantes, incluindo ano, mês, dia da semana, hora e lags das observações anteriores.

3. **Divisão dos Dados em Treinamento e Teste**:
   - Os dados são divididos em conjuntos de treinamento e teste, com 10% do conjunto de dados original sendo usado para teste.

4. **Treinamento de Modelos**:
   - Modelos de regressão linear e CatBoost são treinados.
   - O modelo CatBoostRegressor é otimizado considerando diferentes taxas de aprendizado e profundidades da árvore.

5. **Avaliação de Modelos**:
   - Os modelos são avaliados usando a métrica REQM no conjunto de teste.
   - É identificado o modelo que atende ao requisito de REQM inferior a 48.

## Resultados e Conclusão

O projeto conclui que o modelo CatBoostRegressor atende ao requisito de manter o REQM no conjunto de teste abaixo de 48. No entanto, observa-se uma diferença substancial entre o desempenho do conjunto de treinamento e teste, indicando possível sobreajuste.

Oportunidades de melhoria são destacadas, como otimização de hiperparâmetros, aplicação de técnicas de validação cruzada e consideração de técnicas de regularização para reduzir a diferença entre o desempenho nos conjuntos de treinamento e teste. O projeto enfatiza que o código é um ponto de partida promissor, mas existem áreas para aprimoramentos futuros.

O projeto representa um esforço sólido na previsão de demanda de táxi em aeroportos e fornece uma estrutura para futuros desenvolvimentos.

## Requisitos de Instalação

Para executar o código deste projeto, é necessário ter as seguintes bibliotecas Python instaladas:

- pandas
- statsmodels
- matplotlib
- scikit-learn
- catboost

Você pode instalar essas bibliotecas usando o pip. Por exemplo:

```bash
pip install pandas statsmodels matplotlib scikit-learn catboost
