import numpy as np
import flwr as fl
from flwr.server import ServerConfig
from aes_utils import encrypt_bytes, decrypt_bytes, get_key_iv
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes
from evaluate import evaluate_model
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)
    def forward(self, x):
        return self.linear(x.view(-1, 28 * 28))

class FedAvgAES(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression()

    def initialize_parameters(self, client_manager):
        return super().initialize_parameters(client_manager)

    def aggregate_fit(self, rnd, results, failures):
        key, iv = get_key_iv()
        decrypted_results = []
        shapes = [v.shape for v in self.model.state_dict().values()]
        for client, fit_res in results:
            param_list = parameters_to_ndarrays(fit_res.parameters)
            decrypted_weights = [
                np.frombuffer(
                    decrypt_bytes(w.tobytes(), key, iv), dtype=np.float32
                ).reshape(shape) if w.dtype == np.uint8 else w
                for w, shape in zip(param_list, shapes)
            ]
            new_fit_res = FitRes(
                parameters=ndarrays_to_parameters(decrypted_weights),
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
                status=fit_res.status,
            )
            decrypted_results.append((client, new_fit_res))
        agg_parameters, agg_num_examples = super().aggregate_fit(rnd, decrypted_results, failures)
        if agg_parameters is None:
            return None, None
        agg_ndarrays = parameters_to_ndarrays(agg_parameters)
        # Đánh giá mô hình trên tập test chung
        evaluate_model(agg_ndarrays, round_num=rnd)
        shapes = [w.shape for w in agg_ndarrays]
        encrypted_agg_parameters = [np.frombuffer(encrypt_bytes(w.tobytes(), key, iv), dtype=np.uint8) for w in agg_ndarrays]
        return ndarrays_to_parameters(encrypted_agg_parameters), agg_num_examples

strategy = FedAvgAES(
    fraction_fit=1.0,
    min_fit_clients=1,
    min_available_clients=1,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config=ServerConfig(num_rounds=3),
    )