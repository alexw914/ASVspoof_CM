import torch

def save_embedding(datasets, encoder, hparams, save_file_dict):

        test_asv_emb_dict, test_cm_emb_dict, test_cm_scores_dict = encoder.evaluate(
                test_set=datasets["trl"],
                min_key="eer",
                progressbar= True,
                test_loader_kwargs=hparams["dataloader_options"],
        )
        enrol_asv_emb_dict, enrol_cm_emb_dict, enrol_cm_score_dict = encoder.evaluate(
                test_set=datasets["enrol"],
                min_key="eer",
                progressbar= True,
                test_loader_kwargs=hparams["dataloader_options"],
        )
        torch.save(test_cm_scores_dict, save_file_dict["test_cm_scores_file"])
        torch.save(test_asv_emb_dict, save_file_dict["test_asv_emb_file"])
        torch.save(enrol_asv_emb_dict, save_file_dict["enrol_asv_emb_file"])
        torch.save(test_cm_emb_dict, save_file_dict["test_cm_emb_file"])

def save_scores(datasets, encoder, hparams, save_file_dict, eval_opt="LA"):
        if eval_opt=="2021LA":
                test_cm_scores_dict, test_cm_emb_dict = encoder.evaluate(
                        test_set=datasets["eval_2021"],
                        min_key="eer",
                        progressbar= True,
                        test_loader_kwargs=hparams["dataloader_options"],
                )
                test_cm_scores_progress_dict, test_cm_emb_progress_dict = encoder.evaluate(
                        test_set=datasets["eval_2021_progress"],
                        min_key="eer",
                        progressbar= True,
                        test_loader_kwargs=hparams["dataloader_options"],
                )
        elif eval_opt=="2021DF":
                test_cm_scores_dict, test_cm_emb_dict = encoder.evaluate(
                        test_set=datasets["df_eval"],
                        min_key="eer",
                        progressbar= True,
                        test_loader_kwargs=hparams["dataloader_options"],
                )
                test_cm_scores_progress_dict, test_cm_emb_progress_dict = encoder.evaluate(
                        test_set=datasets["df_progress"],
                        min_key="eer",
                        progressbar= True,
                        test_loader_kwargs=hparams["dataloader_options"],
                )
        elif eval_opt=="2019LA":
                test_cm_scores_dict, test_cm_emb_dict = encoder.evaluate(
                        test_set=datasets["eval_2019"],
                        min_key="eer",
                        progressbar= True,
                        test_loader_kwargs=hparams["dataloader_options"],
                )
                test_cm_scores_progress_dict, test_cm_emb_progress_dict = encoder.evaluate(
                        test_set=datasets["eval_2019_progress"],
                        min_key="eer",
                        progressbar= True,
                        test_loader_kwargs=hparams["dataloader_options"],
                )
        else:
                raise Exception()
        torch.save(test_cm_scores_dict, save_file_dict["test_cm_scores_file"])
        torch.save(test_cm_emb_dict, save_file_dict["test_cm_emb_file"])
        torch.save(test_cm_scores_progress_dict, save_file_dict["test_cm_scores_progress_file"])
        torch.save(test_cm_emb_progress_dict, save_file_dict["test_cm_emb_progress_file"])
