import lightning as L
import os
import torch
import yaml

from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from model.word_classifier import WordClassifier

from train_utils import construct_datasets

def cli_main():

    # Enable usage of tensor cores if available
    torch.set_float32_matmul_precision('high')

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--test_ckpt', default=None, type=str)
    parser.add_argument('--vocab_size', default=250, type=int)
    parser.add_argument('--dset', default='libribrain', type=str)
    parser.add_argument('--aux_dsets', default=[], type=str, nargs='+')
    parser.add_argument('--limit_eval_samples', default=None, type=int)
    parser.add_argument('--context', type=int, default=64)
    parser.add_argument('--tmin', type=float, default=-0.5)
    parser.add_argument('--tmax', type=float, default=2.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--name', type=str, default="word-decoding")
    parser.add_argument('--train_scale', type=float, default=1.0)
    parser.add_argument('--save_train_transcripts', action="store_true", default=False)
    parser.add_argument('--save_train_transcripts_name', type=str, default="train_transcripts.csv")
    
    # Add model specific args
    parser = WordClassifier.add_model_specific_args(parser)
    
    # Add trainer args
    args = parser.parse_args()

    L.seed_everything(args.seed)

    with open("./data/dataset_configs.yaml") as f:
        dataset_configs = yaml.safe_load(f)

    datasets, dataset, channels, subjects, word_embeddings, top_words_map, other_words = construct_datasets(
        args.dset, args.aux_dsets, dataset_configs, args
    )
    print("Top words:", top_words_map.keys())
    
    if args.test_ckpt is None:
        train_loader = DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=False if args.debug else True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    # Limit eval samples to avoid bankruptcy
    if args.limit_eval_samples is not None:
        seed = 42
        generator = torch.Generator().manual_seed(seed)
        num_samples = min(args.limit_eval_samples, len(datasets["test"]))
        if num_samples < args.limit_eval_samples:
            print(f"Warning: Fewer eval samples {num_samples} than requested limit {args.limit_eval_samples}")
        random_sampler = torch.utils.data.RandomSampler(datasets["test"], num_samples=num_samples, generator=generator) 
        test_loader = DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=random_sampler,
        )
    else:
        test_loader = DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )


    # ------------
    # model
    # ------------
    model = WordClassifier(
        n_channels=channels,  # number of input channels
        n_classes=len(top_words_map),  # number of classes
        word_embeddings=word_embeddings,
        top_words_map=top_words_map,
        other_words=other_words,
        n_subjects=len(subjects),
        dataset=dataset,
        **vars(args)
    )

    # ------------
    # training
    # ------------
    ckpt = ModelCheckpoint(
        monitor='val_top10acc',
        dirpath='checkpoints',
        filename=args.name + '-{epoch:02d}-{val_top10acc:.4f}',
        mode='max',
    )
    callbacks = [
        ckpt,
        EarlyStopping(
            monitor='val_top10acc',
            patience=args.patience,
            mode='max'
        ),
    ]

    logger = WandbLogger(
        name=args.name,
        project="word-to-sent",
        log_model=False,
    )

    trainer_params = dict(
        callbacks=callbacks,
        accelerator='auto',
        devices=1,
        logger=logger,
    )

    if args.debug:
        trainer_params.update(
            overfit_batches=1,
            limit_train_batches=1,
            log_every_n_steps=1,
        )

    trainer = L.Trainer(**trainer_params)

    # ------------
    # train & test
    # ------------
    if args.test_ckpt is not None:
        model = WordClassifier.load_from_checkpoint(args.test_ckpt, top_words_map=top_words_map, other_words=other_words, **vars(args))
    else:
        trainer.fit(model, train_loader, val_loader)
        model = WordClassifier.load_from_checkpoint(
            ckpt.best_model_path,
            top_words_map=top_words_map,
            other_words=other_words,
            **vars(args)
        )

    if args.save_train_transcripts:
        # Generate transcripts from train set
        print(f"Generating transcripts from train set and saving to {args.save_train_transcripts_name}")
        model.save_transcripts = True
        model.transcript_output_file = args.save_train_transcripts_name
        trainer.predict(model, train_loader)
        model.save_transcripts = False
        print(f"Transcripts saved to {args.save_train_transcripts_name}")
        return

    if "holdout" in datasets:
        holdout_loader = DataLoader(
            datasets["holdout"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        trainer.test(model, dataloaders={
            'test': test_loader,
            'holdout': holdout_loader
        })
    else:
        trainer.test(model, test_loader)

if __name__ == '__main__':
    cli_main()