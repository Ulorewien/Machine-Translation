def get_config():
    return {
        "batch_size": 32,
        "n_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 64,
        "language_source": "en",
        "language_target": "it",
        "model_folder": "weights",
        "model_basename": "model_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/model",
        "n_layer": 4,
        "n_heads": 2,
        "dropout": 0.1,
        "d_ff": 256,
    }