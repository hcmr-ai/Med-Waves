import xgboost as xgb

class XGBIncremental:
    def __init__(self, rounds_per_batch=10, **params):
        # sensible defaults for bias regression
        default = dict(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",   # memory + speed
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            max_bin=256,
        )
        default.update(params)
        self.params = default
        self.rounds_per_batch = rounds_per_batch
        self.booster = None

    def partial_fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        if self.booster is None:
            self.booster = xgb.train(self.params, dtrain, num_boost_round=self.rounds_per_batch)
        else:
            self.booster = xgb.train(self.params, dtrain, num_boost_round=self.rounds_per_batch, xgb_model=self.booster)

    def predict(self, X):
        if self.booster is None:
            raise RuntimeError("Model not trained yet.")
        return self.booster.predict(xgb.DMatrix(X))

    # for saving alongside your other artifacts
    def save(self, path):
        self.booster.save_model(path)

    @classmethod
    def load(cls, path):
        m = cls()
        m.booster = xgb.Booster()
        m.booster.load_model(path)
        return m
