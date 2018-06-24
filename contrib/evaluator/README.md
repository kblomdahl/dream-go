# `dream_go/evaluator:0.5.0`

This image is responsible for calculating the ELO each network. This is done by playing playoff-style tournaments using `gomill` and then rating the networks on the result.

The image only plays said tournament between the five most recent networks, but the rating is computed globally. So it should get more accurate over time.
