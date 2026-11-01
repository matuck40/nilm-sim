# Tour do código (para estudar, em português)

O código, os nomes e a interface estão em inglês (repositório público). Os aparelhos: fridge = geladeira, light = luz, fan = ventilador, shower = chuveiro.

O dado anda nesta ordem. Leia os arquivos nesta mesma ordem.

## 1. `nilm_sim/appliances.py`: o aparelho
- Cada aparelho é uma lista de estados, e cada estado tem (P, Q). Bloco 1: potência ativa e reativa.
- `render()` transforma "estado a cada segundo" em P(t) e Q(t). É aqui que nasce o pico de partida da geladeira.
- **Experimente:** mude o Q do ventilador para 0 e rode o relatório. O Hart passa a confundir o ventilador com a TV e a luz? Por quê? (Dica: plano P-Q.)

## 2. `nilm_sim/house.py`: a casa
- `default_schedule()` decide quando cada aparelho liga (rotina).
- `simulate_house()` soma tudo com a carga de base e o ruído: o `P` é o medidor, e os `P_<appliance>` são a submedição (gabarito).
- `LiveHouse` faz a mesma coisa segundo a segundo, com os estados vindos dos botões do app.

## 3. `nilm_sim/dataset.py`: o split por casa
- `split_by_house()` separa casas inteiras. Bloco 1B: é o que torna o teste honesto.
- **Experimente:** no app, aba "Simulated day", compare uma casa de treino (semente 0 a 6) com uma não vista (7 em diante).

## 4. `nilm_sim/events.py`: o Hart
- `detect_events()`: filtro de mediana, procura de degraus e medição de (ΔP, ΔQ).
- `learn_signatures()`: aprende a assinatura de cada aparelho pela submedição do treino.
- `HartDisaggregator.predict()`: rotula cada degrau pela assinatura mais próxima e casa o liga com o desliga.
- **Experimente:** baixe `threshold` de 30 para 20 W. O ventilador melhora (a troca de velocidade é um degrau de 25 W)? O que piora?

## 5. `nilm_sim/models.py`: o Random Forest
- `make_features()`: a janela deslizante do Bloco 2. Só usa passado e presente; nada do futuro.
- `stride=5`: treina com 1 a cada 5 segundos, porque segundos vizinhos são quase iguais.
- **Experimente:** `ForestDisaggregator(use_hour=True)`. Melhora? Isso seria vazamento? (Não: a hora do dia é conhecida no momento da previsão.)

## 6. `nilm_sim/metrics.py` e `report.py`: a avaliação
- MAE (W), SAE (energia total) e F1 (ligado ou desligado), por aparelho.
- `report.py` treina nas casas 0 a 6, testa nas 7 a 9 e grava `reports/metrics.md`.

## 7. `app.py`: a interface
- `@st.cache_resource` treina os modelos uma vez só, quando o app sobe.
- `@st.fragment(run_every=1.0)` redesenha só o painel ao vivo a cada segundo, sem recarregar a página.

## Perguntas para treinar em voz alta
1. Por que o Hart ganha na luz e perde no ventilador?
2. O Random Forest prevê potência negativa? Onde isso é tratado?
3. Se você treinasse e testasse na mesma casa, dividindo no tempo, o F1 subiria ou desceria? Por quê?
