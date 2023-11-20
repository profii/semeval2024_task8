import argparse


def get_args():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    return args

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument('--model_name',
                        type=str,
                        default="bert-base-uncased")
    
    parser.add_argument('--seed_val',
                        type=int,
                        default=42)
    
    parser.add_argument('--test_size',
                        type=int,
                        default=0.2)
    
    parser.add_argument('--train_path',
                        type=str,
                        default="data/subtaskC_train.jsonl")
    
    parser.add_argument('--test_path',
                        type=str,
                        default="data/subtaskC_dev.jsonl")
    
    parser.add_argument('--output_dir',
                        type=str,
                        default="experiments/")
    
    parser.add_argument('--logging_dir',
                        type=str,
                        default="experiments/logs")
    
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-5)
    
    parser.add_argument('--num_epochs',
                        type=int,
                        default=4)
    
    parser.add_argument('--evaluation_strategy',
                        type=str,
                        default="epoch")
    
    parser.add_argument('--save_strategy',
                        type=str,
                        default="epoch")
    
    parser.add_argument('--save_total_limit',
                        type=int,
                        default=2)
    
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=4)
    
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=4)
    
    parser.add_argument('--save_steps',
                        type=int,
                        default=100)

    parser.add_argument('--logging_steps',
                        type=int,
                        default=10)

    parser.add_argument('--add_prefix_space',
                        type=bool,
                        default=True)

    parser.add_argument('--max_length',
                        type=int,
                        default=510)

    # parser.add_argument('--',
    #                     type=str,
    #                     default="bert-base-uncased")

    # parser.add_argument('--',
    #                     type=str,
    #                     default="bert-base-uncased")

    # parser.add_argument('--',
    #                     type=str,
    #                     default="bert-base-uncased")
    # parser.add_argument('--',
    #                     type=str,
    #                     default="bert-base-uncased")
    # parser.add_argument('--',
    #                     type=str,
    #                     default="bert-base-uncased")
    # parser.add_argument('--',
    #                     type=str,
    #                     default="bert-base-uncased")
    # parser.add_argument('--',
    #                     type=str,
    #                     default="bert-base-uncased")
    # parser.add_argument('--',
    #                     type=str,
    #                     default="bert-base-uncased")
    # parser.add_argument('--',
    #                     type=str,
    #                     default="bert-base-uncased")
    # parser.add_argument('--',
    #                     type=str,
    #                     default="bert-base-uncased")

