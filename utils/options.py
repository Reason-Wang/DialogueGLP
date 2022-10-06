import argparse


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_basemodel_options(self):
        self.parser.add_argument('--lr', type=float, default=0.002)
        self.parser.add_argument('--plm_lr', type=float, default=1e-5)
        self.parser.add_argument('--hidden_size', type=int, default=1024)
        self.parser.add_argument('--lstm_hidden_size', type=int, default=1024)
        self.parser.add_argument('--n_sentences', type=int, default=2)
        self.parser.add_argument('--n_knowledges', type=int, default=2)
        self.parser.add_argument('--max_len', type=int, default=512)
        self.parser.add_argument('--fc_dropout', type=float, default=0.2)
        self.parser.add_argument('--freeze_epochs', type=int, default=2)


    def add_dialogue_infer_options(self):
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--fc_dropout', type=float, default=0.2)
        self.parser.add_argument('--hidden_size', type=int, default=100)

    def add_dialogue_rnn_options(self):
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--fc_dropout', type=float, default=0.2)
        self.parser.add_argument('--hidden_size', type=int, default=100)

    def add_dialogue_gcn_options(self):
        self.parser.add_argument('--lr', type=float, default=0.0001)
        self.parser.add_argument('--fc_dropout', type=float, default=0.2)
        self.parser.add_argument('--hidden_size', type=int, default=300)

    def add_dialogue_crn_options(self):
        self.parser.add_argument('--base_layer', type=int, default=2)
        self.parser.add_argument('--hidden_size', type=int, default=100)
        self.parser.add_argument('--n_speakers', type=int, default=9)
        self.parser.add_argument('--fc_dropout', type=float, default=0.1)
        self.parser.add_argument('--lr', type=float, default=0.0005)

    def add_extractor_options(self):
        self.parser.add_argument('--lr', type=float, default=2e-5)
        self.parser.add_argument('--max_len', type=int, default=512)
        self.parser.add_argument('--fc_dropout', type=float, default=0.1)
        self.parser.add_argument('--apex', action='store_true')


    def print_options(self):
        pass

    def initialize_parser(self):
        self.parser.add_argument('--name', type=str, help='name of the experiment')
        self.parser.add_argument('--model', type=str, help='name of the model')
        self.parser.add_argument('--dataset', type=str, help='name of the dataset')
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument('--input_size', type=int, default=1024)
        self.parser.add_argument('--metric', type=str, default='weighted')
        self.parser.add_argument('--no_shuffle_train', action='store_true')
        self.parser.add_argument('--target_size', type=int, default=7)
        self.parser.add_argument('--scheduler', type=str, default='cosine')
        self.parser.add_argument('--num_cycles', type=float, default=0.5)
        self.parser.add_argument('--num_workers', type=int, default=0)
        self.parser.add_argument('--epochs', type=int, default=10)
        self.parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
        self.parser.add_argument('--num_warmup_steps', type=int, default=0)
        self.parser.add_argument('--knowledge', type=str, default='none')
        self.parser.add_argument('--feature_metric', type=str, default='macro')
        self.parser.add_argument('--cls_3', action='store_true')
        self.parser.add_argument('--gradient_clipping', action='store_true')
        self.parser.add_argument('--max_grad_norm', type=float, default=1000.0)


    def parse(self):
        opt = self.parser.parse_known_args()
        return opt

    def get_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        return message

