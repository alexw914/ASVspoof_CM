import torch
import speechbrain as sb
from torch.utils.data import DataLoader
from speechbrain import Stage
from tqdm.contrib import tqdm
from pytorch_model_summary import summary
from models.BinaryMetricStats import BinaryMetricStats

class CM(sb.Brain):

    def compute_forward(self, batch, stage):
        
        batch = batch.to(self.device)
        features, lens = self.prepare_features(batch.sig, batch, stage)
        features = features.transpose(1,2)
        enc_output = self.modules.cm_encoder(features)

        return enc_output

    def prepare_features(self, wavs, batch, stage):

        wavs, lens = wavs
        if stage == sb.Stage.TRAIN:
            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):
                # Apply augment
                wavs_aug = augment(batch.id).to(self.device)
                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs
                    
            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = 4
            lens = torch.cat([lens] * self.n_augment)

        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        return feats, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        # Concatenate labels (due to data augmentation)

        bonafide_encoded, lens = batch.bonafide_encoded
        enc_output = predictions

        if stage==sb.Stage.TRAIN:
            
            bonafide_encoded = torch.cat([bonafide_encoded]*self.n_augment, dim = 0)
            lens = torch.cat([lens]*self.n_augment, dim=0)

            # cm_loss, cm_score = self.modules.cm_loss_metric(torch.squeeze(enc_output,1), torch.squeeze(bonafide_encoded,1))
            # attack_loss, acc = self.modules.attack_loss_metric(torch.squeeze(enc_output,1), torch.squeeze(attack_encoded,1))
            # loss = attack_loss + cm_loss
            # self.top1.append(acc.detach().item())

        cm_loss, cm_score = self.modules.cm_loss_metric(torch.squeeze(enc_output,1), torch.squeeze(bonafide_encoded,1))
        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, cm_score, bonafide_encoded)
        return cm_loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up evaluation-only statistics trackers
        # self.top1 = []
        if stage != sb.Stage.TRAIN:
            self.error_metrics = BinaryMetricStats(eval_opt="2021LA-progress")

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            # self.acc = sum(self.top1)/len(self.top1)
        # Summarize the statistics from the stage for record-keeping.
        else:
            self.error_metrics.summarize()
            stats = {
                "loss": stage_loss,
                "eer": self.error_metrics.summary["EER"],
                "min_tDCF": self.error_metrics.summary["min_tDCF"]
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_scheduler(current_epoch = epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats,
                                                 num_to_keep=5,
                                                 min_keys=["eer"],
                                                 keep_recent=False
                                                 )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def evaluate_batch(self, batch, stage):
        """
        Overwrite evaluate_batch.
        Keep same for stage in (TRAIN, VALID)
        Output probability in TEST stage (from classify_batch)
        """

        if stage != sb.Stage.TEST:
            # Same as before
            out = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(out, batch, stage=stage)
            return loss.detach().cpu()
        else:
            enc_output = self.compute_forward(batch, stage=stage)
            cm_loss, cm_score =  self.modules.cm_loss_metric(torch.squeeze(enc_output,1), is_train=False)
            return cm_score, enc_output

    def evaluate(
            self,
            test_set,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
    ):
        """
        Overwrite evaluate() function so that it can output score file
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()

        """
        added here
        """
        cm_score_dict = {}
        cm_emb_dict = {}
        with torch.no_grad():
            for batch in tqdm(
                    test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                """
                Rewrite here
                """
                cm_scores, cm_emb = self.evaluate_batch(batch, stage=Stage.TEST)
                cm_scores = [cm_scores[i].item() for i in range(cm_scores.shape[0])]
                cm_emb = cm_emb.unsqueeze(1)
                for i, seg_id in enumerate(batch.id):
                    cm_emb_dict[seg_id] = cm_emb[i].detach().clone()
                    cm_score_dict[seg_id] = cm_scores[i]

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

        self.step = 0
        return cm_score_dict, cm_emb_dict


if __name__ =="__main__":
    
    # os.environ["CUDA_VISABLE_DEVICE"] = "1"
    # # TDNN = CM_Decoder(input_shape=[ None, None,256],out_neurons=2)
    # TDNN = VC_Discriminator(input_size=256,lin_neurons=512,out_neurons=4)
    # print(summary(TDNN, torch.randn((64,256)), show_input=False))
    pass
