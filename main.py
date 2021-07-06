from src.parser import main_parser
from src.processor import processor
from src.utils import timed_main, set_seed


@timed_main(use_git=True)
def main():
    # Parse input parameters and save/load config file
    args = main_parser()

    # Set seed for reproducibility
    if args.reproducibility:
        set_seed(seed_value=12345, use_cuda=args.use_cuda)

    # Initialize pre-process, data-loader and model
    trainer = processor(args)

    if args.phase == 'train':
        trainer.train()
    elif args.phase == 'test':
        trainer.test()
    elif args.phase == 'train-test':
        trainer.train_test()
    elif args.phase == 'pre-process':
        print("Data pre-processing finished.")
    else:
        raise ValueError(
            "Unsupported phase! args.phase can only take the following values: "
            "'train', 'test', 'train-test' or 'pre-process'")


if __name__ == '__main__':
    main()
