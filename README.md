# Predição de Tráfego Urbano em Semáforos

Sistema de predição de tráfego urbano utilizando redes neurais GRU (Gated Recurrent Unit) para análise e previsão do fluxo de veículos em quatro cruzamentos diferentes.

## Sobre o Projeto

Este projeto utiliza técnicas de aprendizado de máquina para prever o tráfego em semáforos urbanos, com o objetivo de auxiliar no planejamento de infraestrutura e otimização do fluxo de veículos. O sistema analisa dados históricos de tráfego de quatro cruzamentos diferentes e gera predições utilizando redes neurais recorrentes.

### Características Principais

- Análise de séries temporais de tráfego urbano
- Predição utilizando redes neurais GRU
- Tratamento individualizado para cada cruzamento
- Normalização e diferenciação de dados para estacionariedade
- Geração automática de gráficos de predição
- Salvamento de modelos treinados para reutilização

## Requisitos

### Dependências do Sistema

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Bibliotecas Python

As seguintes bibliotecas são necessárias:

- numpy - Operações numéricas e arrays
- pandas - Manipulação de dados
- matplotlib - Visualização de dados
- seaborn - Visualizações estatísticas
- tensorflow - Framework de deep learning
- statsmodels - Testes estatísticos
- scikit-learn - Métricas de avaliação

## Instalação

### 1. Clone o Repositório

```bash
git clone <url-do-repositorio>
cd traffic-lights
```

### 2. Crie um Ambiente Virtual (Recomendado)

```bash
python3 -m venv venv
```

### 3. Ative o Ambiente Virtual

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Instale as Dependências

```bash
pip install -r requirements.txt
```

## Estrutura dos Dados

O projeto utiliza o arquivo `traffic.csv` que deve conter as seguintes colunas:

- `DateTime`: Data e hora da medição (formato: YYYY-MM-DD HH:MM:SS)
- `Junction`: Identificador do cruzamento (1, 2, 3 ou 4)
- `Vehicles`: Número de veículos registrados
- `ID`: Identificador único da medição

### Exemplo de Dados

```csv
DateTime,Junction,Vehicles,ID
2015-11-01 00:00:00,1,15,20151101001
2015-11-01 01:00:00,1,13,20151101011
2015-11-01 00:00:00,2,8,20151101002
```

## Como Executar

### Execução Básica

```bash
python traffic-prediction-gru.py
```

### O que o Script Faz

1. **Carrega os dados** do arquivo `traffic.csv`
2. **Realiza análise exploratória** dos dados
3. **Processa e transforma** os dados para cada cruzamento:
   - Junction 1: Diferenciação semanal (24×7 horas)
   - Junction 2: Diferenciação diária (24 horas)
   - Junction 3 e 4: Diferenciação horária (1 hora)
4. **Treina modelos GRU** para cada cruzamento
5. **Gera predições** e calcula métricas de erro (RMSE)
6. **Salva automaticamente**:
   - Modelos treinados (`.h5`)
   - Gráficos de predição (`.png`)
   - Gráficos de comparação (`.png`)

### Arquivos Gerados

Após a execução, os seguintes arquivos serão criados:

**Modelos Treinados:**
- `Junction1_GRU_model.h5`
- `Junction2_GRU_model.h5`
- `Junction3_GRU_model.h5`
- `Junction4_GRU_model.h5`

**Gráficos de Predição:**
- `Junction1_GRU_predictions.png`
- `Junction2_GRU_predictions.png`
- `Junction3_GRU_predictions.png`
- `Junction4_GRU_predictions.png`

**Gráficos de Comparação:**
- `Junction1_comparison.png`
- `Junction2_comparison.png`
- `Junction3_comparison.png`
- `Junction4_comparison.png`

## Metodologia

### 1. Análise Exploratória de Dados (EDA)

- Análise de tendências e sazonalidade
- Extração de características temporais (ano, mês, dia, hora, dia da semana)
- Visualização de padrões de tráfego

### 2. Pré-processamento

**Normalização:**
```python
valor_normalizado = (valor - média) / desvio_padrão
```

**Diferenciação:**
- Aplicada para remover tendências e sazonalidade
- Intervalos diferentes para cada cruzamento baseado em suas características

### 3. Arquitetura do Modelo GRU

```
Camada 1: GRU (64 unidades) + Dropout (0.2)
Camada 2: GRU (32 unidades) + Dropout (0.2)
Camada 3: GRU (16 unidades) + Dropout (0.2)
Camada 4: Dense (1 unidade)
```

**Parâmetros de Treinamento:**
- Otimizador: SGD (learning_rate=0.01, momentum=0.9)
- Função de perda: Mean Squared Error
- Épocas: 15 (com early stopping)
- Batch size: 150
- Divisão: 90% treino, 10% teste

### 4. Avaliação

- Métrica principal: RMSE (Root Mean Squared Error)
- Comparação visual entre valores reais e preditos
- Inversão das transformações para escala original

## Interpretação dos Resultados

### RMSE (Root Mean Squared Error)

O RMSE indica o erro médio das predições. Quanto menor o valor, melhor o modelo.

### Gráficos de Predição

- **Linha colorida**: Valores reais do conjunto de teste
- **Linha cinza**: Valores preditos pelo modelo

### Gráficos de Comparação

Mostram lado a lado as predições e os valores originais na escala real (não transformada).

## Características dos Cruzamentos

### Junction 1
- Tendência ascendente forte
- Sazonalidade semanal pronunciada
- Picos durante manhã e tarde

### Junction 2
- Tendência ascendente moderada
- Pico de tráfego em junho
- Padrão diário consistente

### Junction 3
- Tendência linear
- Tráfego mais estável
- Variação horária predominante

### Junction 4
- Dados limitados (apenas 2017)
- Padrão horário
- Dados esparsos

## Tempo de Execução

O treinamento completo dos quatro modelos leva aproximadamente:
- **5-10 minutos** em hardware moderno com GPU
- **15-30 minutos** em CPU

## Solução de Problemas

### Erro: "FileNotFoundError: traffic.csv"

**Solução:** Certifique-se de que o arquivo `traffic.csv` está no mesmo diretório do script.

### Erro: "ModuleNotFoundError"

**Solução:** Instale as dependências:
```bash
pip install -r requirements.txt
```

### Erro de Memória

**Solução:** Reduza o batch_size no código ou use um computador com mais RAM.

### Aviso: "Your CPU supports instructions that this TensorFlow binary was not compiled to use"

**Solução:** Este é apenas um aviso de performance. O código funcionará normalmente, mas pode ser mais lento. Para melhor performance, instale o TensorFlow otimizado para seu processador.

## Melhorias Futuras

- Implementação de LSTM e Bidirectional LSTM para comparação
- Incorporação de dados externos (clima, eventos, feriados)
- API REST para predições em tempo real
- Dashboard interativo para visualização
- Otimização de hiperparâmetros com Grid Search
- Predição de múltiplos passos à frente

## Contribuindo

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Referências

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Time Series Forecasting](https://machinelearningmastery.com/time-series-forecasting/)
- [GRU Networks](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
- [Statsmodels - Augmented Dickey-Fuller Test](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)

## Contato

Para dúvidas, sugestões ou reportar problemas, abra uma issue no repositório.

---

**Desenvolvido para análise e predição de tráfego urbano utilizando Deep Learning**

