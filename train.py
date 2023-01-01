import os,sys,torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from dataset.cm_eval_dataset import get_eval_dataset
from models.cm_2021_df import CM_DF
from models.cm_2021 import CM
# from models.cm_2021_ecapa import CM   # using codec and transmission using this model
from models.cm_2019 import CM_2019
from dataset.cm_dataset import get_dataset
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    torch.cuda.empty_cache()

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin,overrides)
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets = get_dataset(hparams)
    evalsets = get_eval_dataset(hparams)

    if hparams["train_option"]=="2021LA":
        cm_model = CM(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )
        cm_model.fit(
            epoch_counter=cm_model.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=evalsets["eval_2021_progress"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["valid_dataloader_options"],
        )
    elif hparams["train_option"]=="2019LA":
        cm_model = CM_2019(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
        cm_model.fit(
            epoch_counter=cm_model.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=evalsets["eval_2019"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["valid_dataloader_options"],
        )
    elif hparams["train_option"]=="2021DF":
        cm_model = CM_DF(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
        cm_model.fit(
        epoch_counter=cm_model.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=evalsets["df_progress"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
        )