# from accelerate import Accelerator
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0
        # self.accelerator = Accelerator()

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        # unwrap_model = self.accelerator.unwrap_model(model)
        # unwrap_ema_model = self.accelerator.unwrap_model(ema_model)
        # unwrap_ema_model.load_state_dict(unwrap_model.state_dict())
        # ema_model = self.accelerator.prepare(unwrap_ema_model)
        # print(type(ema_model), type(model))
        ema_model.load_state_dict({k.replace("module.", ""): v for k, v in model.state_dict().items()})

# from accelerate import Accelerator

# class EMA:
#     def __init__(self, beta):
#         super().__init__()
#         self.beta = beta
#         self.step = 0

#     def update_model_average(self, ma_model, current_model):
#         for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
#             old_weight, up_weight = ma_params.data, current_params.data
#             ma_params.data = self.update_average(old_weight, up_weight)

#     def update_average(self, old, new):
#         if old is None:
#             return new
#         return old * self.beta + (1 - self.beta) * new

#     def step_ema(self, ema_model, model, step_start_ema=2000):
#         accelerator = Accelerator()

#         if self.step < step_start_ema:
#             self.reset_parameters(ema_model, model)
#             self.step += 1
#             return

#         model, ema_model = accelerator.prepare(model, ema_model)

#         with accelerator.start_scaling_loss():
#             self.update_model_average(ema_model, model)

#         self.step += 1

#     def reset_parameters(self, ema_model, model):
#         ema_model.load_state_dict(model.state_dict())
