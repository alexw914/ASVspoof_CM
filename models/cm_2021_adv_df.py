import torch,librosa,torchaudio,nnAudio.Spectrogram
import speechbrain as sb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from speechbrain import Stage
from tqdm.contrib import tqdm
from pytorch_model_summary import summary
from models.BinaryMetricStats import BinaryMetricStats
from speechbrain.nnet.losses import nll_loss

class CM(sb.Brain):

    def compute_forward(self, batch, stage):
        
        batch = batch.to(self.device)
        lfccs, lens = self.prepare_features(batch.sig, batch, stage)
        lfccs = lfccs.transpose(1,2)
        enc_output = self.modules.cm_encoder(lfccs)
        codec_output = self.modules.domain_classifier(enc_output)

        return enc_output,codec_output

    def prepare_features(self, wavs, batch, stage):

        wavs, lens = wavs
        feats = self.modules.compute_lfcc(wavs)
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
        enc_output,codec_output = predictions

        cm_loss, cm_score = self.modules.cm_loss_metric(torch.squeeze(enc_output,1), torch.squeeze(bonafide_encoded,1))
        # Compute classification error at test time
        loss = cm_loss
        if stage == sb.Stage.TRAIN:
            domain_encoded, lens = batch.domain_encoded
            codec_output = F.log_softmax(codec_output,dim=1)
            codec_loss = nll_loss(codec_output,torch.squeeze(domain_encoded,1))
            loss = cm_loss + codec_loss*0.0001
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, cm_score, bonafide_encoded)
        return loss

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
            self.error_metrics = BinaryMetricStats(eval_opt="2021DF-progress")

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
                                                 min_keys=["eer","min_tDCF"],
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
            enc_output, codec_output = self.compute_forward(batch, stage=stage)
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
        avg_test_loss = 0.0

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