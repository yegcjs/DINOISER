import torch
import torch.nn as nn



class LengthPredictor(nn.Module):
    def __init__(self, args, unused=None) -> None:
        super().__init__()
        self.max_length = args.max_target_positions
        self.predict_type = args.length_predict_type

    def get_feature(self, enc_out):
        # mean pooling
        feature, padding_mask = enc_out["feature"].detach(), enc_out["padding_mask"]
        assert feature.shape[:2] == padding_mask.shape
        feature = feature.masked_fill(padding_mask.unsqueeze(-1), 0).sum(dim=1)
        src_length = padding_mask.shape[1] - padding_mask.sum(-1)
        feature = feature / src_length[:, None]
        return feature
    
    def get_prediction(self, model_predict_length, enc_out):    # post process
        if self.predict_type == "fixed":
            batch_size, device = enc_out["feature"].size(0), enc_out["feature"].device
            length = self.max_length * torch.ones((batch_size, ), dtype=torch.long, device=device)
        elif self.predict_type == "absolute":
            length = model_predict_length
        elif self.predict_type == "difference":
            padding_mask = enc_out["padding_mask"]
            src_length = padding_mask.shape[1] - padding_mask.sum(-1)
            length = src_length[:, None] + (model_predict_length - self.max_length // 2)
        else:
            raise NotImplementedError
        return length.clamp(2, self.max_length)

    def preprocess_target(self, target_length, enc_out):
        if self.predict_type == "fixed":
            batch_size, device = enc_out["feature"].size(0), enc_out["feature"].device
            return self.max_length * torch.ones((batch_size, ), dtype=torch.long, device=device)
        elif self.predict_type == "absolute":
            return target_length 
        elif self.predict_type == "difference":
            padding_mask = enc_out["padding_mask"]
            src_length = padding_mask.shape[1] - padding_mask.sum(-1)
            return target_length - src_length + self.max_length // 2
        else:
            raise NotImplementedError

    def forward(self, enc_out, length_beam=1):
        return {
            "prediction": self.get_prediction(None, enc_out)
        }

class LengthClassifier(LengthPredictor):
    def __init__(self, args, hidden_dim) -> None:
        super().__init__(args)
        self.length_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, args.max_target_positions)
        )

    def preprocess_target(self, target_length, enc_out):
        target_length = super().preprocess_target(target_length, enc_out)
        return target_length.clamp(0, self.max_length - 1)

    def get_prediction(self, model_out, enc_out, length_beam=1):
        prediction = model_out.topk(dim=-1, k=length_beam).indices
        return super().get_prediction(prediction, enc_out)
   
    def forward(self, enc_out, length_beam=1):
        assert enc_out is not None
        feature = self.get_feature(enc_out)
        model_out = self.length_predictor(feature)
        return {
            "model_out": model_out,
            "prediction": self.get_prediction(model_out, enc_out, length_beam=length_beam)
        }

class LengthRegressor(LengthPredictor):
    def __init__(self, args, hidden_dim) -> None:
        super().__init__(args)
        self.length_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def preprocess_target(self, target_length, enc_out):
        target_length = super().preprocess_target(target_length, enc_out)
        return ((target_length / self.max_length) - 1) * 2     # scaled to -1, 1

    def get_prediction(self, model_out, enc_out, length_beam=1):
        scaled_model_out = ((model_out.clamp(-1, 1) / 2 + 1) * self.max_length).long()
        scaled_model_out = scaled_model_out - (length_beam - 1) // 2
        scaled_model_out = scaled_model_out.masked_fill(scaled_model_out < 0, 0)
        i = torch.arange(length_beam).expand(scaled_model_out.shape[0], -1)
        scaled_model_out = i + scaled_model_out
        return super().get_prediction(scaled_model_out, enc_out)

    def forward(self, enc_out, length_beam=1):
        assert enc_out is not None
        feature = self.get_feature(enc_out)
        model_out = self.length_predictor(feature).squeeze(-1)
        
        return {
            "model_out": model_out,
            "prediction": self.get_prediction(model_out, enc_out, length_beam=length_beam)
        }
    



LENGTH_PREDICTORS = {
    "none": LengthPredictor,
    "classification": LengthClassifier,
    "regression": LengthRegressor
}