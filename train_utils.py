import torch

from data import dataset as ds
from model.word_embeddings import generate_word_embeddings

def construct_datasets(dset, aux_dsets, config, args):

    root = config[dset]["root"]

    # Compute embedding targets
    top_words_map, other_words = ds.find_top_words(root, args.vocab_size)
    other_words = [w for w in other_words if len(w) > 2]
    word_embeddings = generate_word_embeddings(top_words_map.keys(), dset, args.vocab_size)
    print(f"Computed top {len(top_words_map)} words; {len(other_words)} other words")

    datasets = {}
    # Construct training datasets
    dataset_names = [dset] + aux_dsets
    dsets = "_".join(dataset_names)
    torch_train_datasets = []
    subject_id_shift = 0
    all_subjects = []
    all_channels = []
    for dataset_name in dataset_names:

        # Datasets can be made of multiple train set configurations
        train_sets = config[dataset_name]["train"]
        if not isinstance(train_sets, list):
            train_sets = [config[dataset_name]["train"]]
        
        for train_set in train_sets:

            if args.test_ckpt is None or args.predict_oov:
                print(f"Loading train set for {dataset_name}")

                if dset == "armeni2022" and dataset_name == "libribrain":
                    if "Sherlock3" in train_set["tasks"]:
                        # Sherlock3 contains "The Adventures of Sherlock Holmes" which Armeni uses.
                        # Specifically, session 9 and 10 are The Adventure of the Engineer's Thumb and The Adventure of
                        # The Noble Bachelor, which are in the val and test sets of Armeni. We remove these here.
                        train_set["sessions"].remove("9")
                        train_set["sessions"].remove("10")

                torch_dataset = ds.MEGDataset(
                    bids_root=config[dataset_name]["root"],
                    save_root=config[dataset_name]["cache"],
                    subjects=train_set["subjects"],
                    sessions=train_set["sessions"],
                    tasks=train_set["tasks"],
                    dataset=dataset_name,
                    top_words_map=top_words_map,
                    context=args.context,
                    overlap=args.context // 2,
                    tmin=args.tmin,
                    tmax=args.tmax,
                    subject_id_shift=subject_id_shift,
                    debug=args.debug,
                )
                torch_train_datasets.append(torch_dataset)
        
        all_subjects.extend(train_sets[0]["subjects"])
        all_channels.append(config[dataset_name]["channels"])
        subject_id_shift += len(train_sets[0]["subjects"])
    
    max_channels = max(all_channels)

    if args.test_ckpt is None or args.predict_oov:
        datasets["train"] = torch.utils.data.ConcatDataset(torch_train_datasets)

        if args.train_scale < 1.0:
            dataset_size = len(datasets["train"])
            n_subset_samples = int(dataset_size * args.train_scale)
            indices = torch.randperm(dataset_size)[:n_subset_samples].tolist()
            datasets["train"] = torch.utils.data.Subset(datasets["train"], indices)
            print(f"Subset train set to {n_subset_samples} samples from {dataset_size} total samples")

        print(f"Loading val set for {dset}")

        datasets["val"] = ds.MEGDataset(
            bids_root=config[dset]["root"],
            save_root=config[dset]["cache"],
            subjects=config[dset]["val"]["subjects"],
            sessions=config[dset]["val"]["sessions"],
            tasks=config[dset]["val"]["tasks"],
            dataset=dset,
            top_words_map=top_words_map,
            context=args.context,
            overlap=0,
            tmin=args.tmin,
            tmax=args.tmax,
            debug=args.debug,
        )

    print(f"Loading test set for {dset}")

    datasets["test"] = ds.MEGDataset(
        bids_root=config[dset]["root"],
        save_root=config[dset]["cache"],
        subjects=config[dset]["test"]["subjects"],
        sessions=config[dset]["test"]["sessions"],
        tasks=config[dset]["test"]["tasks"],
        dataset=dset,
        top_words_map=top_words_map,
        context=args.context,
        overlap=0,
        tmin=args.tmin,
        tmax=args.tmax,
        debug=args.debug,
    )

    return datasets, dsets, max_channels, all_subjects, word_embeddings, top_words_map, other_words
