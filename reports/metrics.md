# Metrics on unseen houses

Training: 7 houses; test: houses [7, 8, 9] (one day each, 1 Hz). Mean over the test houses.

## Learned signatures (Hart)

| appliance   |     dP |   dQ |   n_events |
|:------------|-------:|-----:|-----------:|
| fridge      |  152.4 | 91.1 |        225 |
| tv          |   93.7 | 14.3 |         12 |
| light       |   60.6 |  0   |         15 |
| fan         |   72.4 | 54.4 |         12 |
| shower      | 5120.2 |  0   |         13 |

## Results

| appliance   |   MAE_W Hart |   MAE_W RandomForest |   SAE Hart |   SAE RandomForest |   F1 Hart |   F1 RandomForest |
|:------------|-------------:|---------------------:|-----------:|-------------------:|----------:|------------------:|
| fan         |        2.718 |                1.12  |      0.359 |              0.053 |     0.667 |             0.977 |
| fridge      |        3.461 |                3.498 |      0.062 |              0.04  |     1     |             0.985 |
| light       |        1.536 |                4.552 |      0.158 |              0.117 |     0.919 |             0.798 |
| shower      |       10.845 |                1.592 |      0.117 |              0.013 |     0.993 |             1     |
| tv          |        7.484 |                5.324 |      0.346 |              0.278 |     0.825 |             0.845 |
