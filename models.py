
import os
import torch
import wandb
import common
import datasets
import data_utils 
import numpy as np
import transformers
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 


class SpeechRecognitionModel(nn.Module):

    def __init__(self, processor):
        super().__init__()
        self.model = transformers.Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h",
            gradient_checkpointing = True,
            ctc_loss_reduction = "mean",
            pad_token_id = processor.tokenizer.pad_token_id
        )
        self.model.freeze_feature_extractor()

    def forward(self, inputs, input_attention, targets): 
        output = self.model(input_values=inputs, attention_mask=input_attention, labels=targets)
        return output.loss, output.logits


class Trainer:

    def __init__(self, args):
        self.args = args
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args)
        self.train_loader, self.val_loader = data_utils.get_dataloaders(
            batch_size = self.config["data"]["batch_size"], read_limit = self.config["data"]["read_limit"])
        
        self.model = SpeechRecognitionModel(processor=self.train_loader.processor).to(self.device)
        self.optim = optim.SGD(self.model.parameters(), lr=self.config["model"]["optim_lr"], weight_decay=0.005, momentum=0.9)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.config["epochs"]-self.config["warmup_epochs"], eta_min=0.0, last_epoch=-1)
        self.warmup_epochs = self.config.get("warmup_epochs", 0)

        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["model"]["optim_lr"] - 1e-12) / self.warmup_epochs 

        self.done_epochs = 1
        self.metric_best = np.inf 
        run = wandb.init(project="hyperverge-asr-prototype")
        self.logger.write(f"Wandb run: {run.get_url()}", mode='info')

        if args["load"] is not None:
            self.load_model(args["load"])

    def compute_word_error_rate(self, loader):
        wer_values, preds, trgs = [], [], []
        metric = datasets.load_metric("wer")
        for idx in range(len(loader)):
            inputs, input_mask, targets = loader.flow()
            inputs, input_mask, targets = inputs.to(self.device), input_mask.to(self.device), targets.to(self.device)
            with torch.no_grad():
                loss, logits = self.model(inputs, input_mask, targets)

            predictions = F.softmax(logits, dim=-1).argmax(dim=-1).detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            targets[targets == -100] = self.train_loader.processor.tokenizer.pad_token_id
            pred_str = self.train_loader.processor.batch_decode(predictions)
            target_str = self.train_loader.processor.batch_decode(targets, group_tokens=False)
            wer_values.append(metric.compute(predictions=pred_str, references=target_str))
            preds.extend(pred_str), trgs.extend(target_str)
            common.progress_bar(status="", progress=(idx+1)/len(loader))
            
        common.progress_bar(status="[WER] {:.4f}".format(np.mean(wer_values)), progress=1.0)
        return np.mean(wer_values), preds, trgs

    def train_on_batch(self, batch):
        self.model.train()
        inputs, input_mask, targets = batch 
        inputs, input_mask, targets = inputs.to(self.device), input_mask.to(self.device), targets.to(self.device)
        loss, logits = self.model(inputs, input_mask, targets)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step() 
        return {"CTC loss": loss.item()}

    def infer_on_batch(self, batch):
        self.model.eval()
        inputs, input_mask, targets = batch 
        inputs, input_mask, targets = inputs.to(self.device), input_mask.to(self.device), targets.to(self.device)
        with torch.no_grad():
            loss, logits = self.model(inputs, input_mask, targets)
        return {"CTC loss": loss.item()}

    def save_model(self, epoch, metric):
        state = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metric": metric,
            "epoch": epoch 
        }    
        torch.save(state, os.path.join(self.output_dir, "best_model.pt"))

    def load_model(self, path):
        if not os.path.exists(os.path.join(self.args["load"], "best_model.pt")):
            raise NotImplementedError(f"Could not find saved model 'best_model.pt' at {self.args['load']}")
        else:
            state = torch.load(os.path.join(self.args["load"], "best_model.pt"), map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.optim.load_state_dict(state["optim"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.done_epochs = state["epoch"]
            self.logger.show(f"Successfully loaded model from {path}", mode='info')

    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group["lr"] = 1e-12 + epoch * self.warmup_rate
        else:
            self.scheduler.step()

    def get_test_performance(self):
        test_meter = common.AverageMeter()
        for idx in range(len(self.val_loader)):
            batch = self.val_loader.flow()
            test_metrics = self.infer_on_batch(batch)
            test_meter.add(test_metrics)
            common.progress_bar(status=test_meter.return_msg(), progress=(idx+1)/len(self.val_loader))

        common.progress_bar(status=test_meter.return_msg(), progress=1.0)
        self.logger.record("Computing WER", mode='test')
        test_wer, preds, trgs = self.compute_word_error_rate(self.val_loader)
        self.logger.record(test_meter.return_msg() + " [WER] {:.4f}".format(test_wer), mode="test")
        print("\n\nSample predictions")
        print("============================================================")
        for i in np.random.choice(np.arange(len(preds)), size=10, replace=False):
            print("Target     : {}".format(trgs[i]))
            print("Prediction : {}".format(preds[i]))
            print("--------------------------------------------------------")

    def train(self):
        print() 
        for epoch in range(max(1, self.done_epochs), self.config["epochs"]+1):
            self.logger.record(f"Epoch {epoch}/{self.config['epochs']}", mode="train")
            train_meter = common.AverageMeter()

            for idx in range(len(self.train_loader)):
                batch = self.train_loader.flow()
                train_metrics = self.train_on_batch(batch)
                train_meter.add(train_metrics)
                wandb.log({"Train CTC loss": train_metrics["CTC loss"]})
                common.progress_bar(status=train_meter.return_msg(), progress=(idx+1)/len(self.train_loader))

            common.progress_bar(status=train_meter.return_msg(), progress=1.0)
            self.logger.record(f"Epoch {epoch}/{self.config['epochs']} Computing WER", mode='train')
            train_wer, _, _ = self.compute_word_error_rate(self.train_loader)
            wandb.log({"Train WER": train_wer, "Epoch": epoch})
            self.logger.write(train_meter.return_msg() + f" [WER] {round(train_wer, 4)}", mode="train")
            self.adjust_learning_rate(epoch)        

            if epoch % self.config["eval_every"] == 0:
                self.logger.record(f"Epoch {epoch}/{self.config['epochs']}", mode='val')
                val_meter = common.AverageMeter()

                for idx in range(len(self.val_loader)):
                    batch = self.val_loader.flow()
                    val_metrics = self.infer_on_batch(batch)
                    val_meter.add(val_metrics)
                    common.progress_bar(status=val_meter.return_msg(), progress=(idx+1)/len(self.val_loader))

                common.progress_bar(status=val_meter.return_msg(), progress=1.0)
                self.logger.record(f"Epoch {epoch}/{self.config['epochs']} Computing WER", mode='val')
                val_wer, _, _ = self.compute_word_error_rate(self.val_loader)
                wandb.log({"Val CTC loss": val_meter.return_metrics()["CTC loss"], "Val WER": val_wer, "Epoch": epoch})
                self.logger.write(val_meter.return_msg() + f" [WER] {round(val_wer, 4)}", mode='val')

                if val_wer < self.metric_best:
                    self.metric_best = val_wer
                    self.save_model(epoch, val_wer)

        print()
        self.logger.record("Training complete! Generating test predictions...", mode='info')    
