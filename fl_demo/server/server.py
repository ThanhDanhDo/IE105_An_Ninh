import flwr as fl
from flwr.server import ServerConfig
from flwr.common import parameters_to_ndarrays
from evaluate import evaluate_model

class FedAvgWithEval(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        agg_parameters, agg_num_examples = super().aggregate_fit(rnd, results, failures)
        if agg_parameters is not None:
            weights = parameters_to_ndarrays(agg_parameters)
            evaluate_model(weights, round_num=rnd)
        return agg_parameters, agg_num_examples

strategy = FedAvgWithEval()

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config=ServerConfig(num_rounds=3),
    )